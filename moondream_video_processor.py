import modal
import os
import time
import tempfile
import shutil
import requests
import ffmpeg # ffmpeg-python - ensures ffmpeg-python is imported as 'ffmpeg'
import cv2 # Still needed for some basic video info if probe fails, though FFmpeg is preferred for main extraction
import numpy as np # Still needed by OpenCV/PIL if you were using them elsewhere, kept for safety
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any, Tuple
import io
import json
import uuid # For generating unique job IDs
import urllib.parse # Added for URL parsing robustness

# --- Constants ---
MODEL_ID = "vikhyatk/moondream2"
REVISION = "2024-05-20"
MODEL_CACHE_PATH = "/model_cache"

# --- Modal Image Definition (Same as before) ---
def download_model_assets():
    from huggingface_hub import snapshot_download as hf_snapshot_download
    hf_snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir=MODEL_CACHE_PATH,
        local_dir_use_symlinks=False,
        cache_dir="/tmp/hf_cache_during_build"
    )

moondream_image = (
    modal.Image.debian_slim(python_version="3.10")
    .run_commands(
        # Install PyTorch with CUDA support specifically
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    )
    .pip_install(
        "transformers==4.39.3", 
        "Pillow==10.0.0", 
        "opencv-python-headless==4.8.0.76", # Kept for probe fallback/compat
        "numpy==1.24.0", 
        "requests==2.28.0", 
        "ffmpeg-python==0.2.0", 
        "fastapi",
        "einops",  # Required by Moondream model
        "pillow",
        "einops",
        "pyvips-binary", # Kept as they were in your original list
        "pyvips",        # Kept as they were in your original list
        "accelerate"
    )
    .apt_install("ffmpeg") # System ffmpeg is crucial for ffmpeg-python method
    .run_function(download_model_assets)
)

app = modal.App(
    "moondream-video-processor-job", # Changed app name slightly
    image=moondream_image
)

# Persistent storage for job status and results
# This Dict will persist across Modal app runs if 'create_if_missing=True'
# and the app name remains the same when deployed.
job_store = modal.Dict.from_name("moondream_video_job_store", create_if_missing=True)


# --- Video Processing Utilities (Updated) ---

# --- Updated VideoDownloader for robust URL handling ---
class VideoDownloader:
    def __init__(self):
        # Create a unique subdirectory within the system's temp directory for this downloader instance
        self.temp_dir_parent = tempfile.mkdtemp(prefix="video_processing_root_")
        self.temp_dir = os.path.join(self.temp_dir_parent, "downloads")
        os.makedirs(self.temp_dir, exist_ok=True)

    def download(self, video_url: str) -> Tuple[str, float]:
        start_time = time.time()
        try:
            # Parse URL and get just the path component for filename purposes
            parsed_url = urllib.parse.urlparse(video_url)
            path_part = parsed_url.path
            
            # Get just the base filename without query parameters for storage
            original_basename = os.path.basename(path_part)
            
            # Generate a safe filename
            name_candidate = ""
            ext_candidate = ".mp4"  # Default extension

            if original_basename:
                name_part, ext_part = os.path.splitext(original_basename)
                if name_part:  # Use name part if it exists
                    name_candidate = name_part
                if ext_part and len(ext_part) > 1:  # Use extension if valid
                    ext_candidate = ext_part
            
            if not name_candidate:  # Fallback if no usable name from URL path
                name_candidate = "video_download"

            # Sanitize and shorten the name part
            safe_name_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            sanitized_name_part = "".join(c for c in name_candidate if c in safe_name_chars)
            if not sanitized_name_part:  # If all chars were stripped
                sanitized_name_part = "video"
            sanitized_name_part = sanitized_name_part[:30]  # Limit length of the name segment

            # Ensure extension is simple and common
            safe_ext = ".mp4"  # Default
            if ext_candidate.startswith(".") and \
               all(c.isalnum() for c in ext_candidate[1:]) and \
               1 < len(ext_candidate) < 6:
                safe_ext = ext_candidate.lower()
            
            # Final filename: sanitized name + short UUID + safe extension
            fname = f"{sanitized_name_part}_{uuid.uuid4().hex[:8]}{safe_ext}"
            
            video_filename = os.path.join(self.temp_dir, fname)

            print(f"Attempting to download {video_url} to {video_filename}")
            # Use the full URL (with query parameters) for the actual download
            with requests.get(video_url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(video_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192 * 4):
                        f.write(chunk)
            
            download_time = time.time() - start_time
            print(f"Download complete in {download_time:.2f}s.")
            return video_filename, download_time
        except Exception as e:
            print(f"Error during video download: {e}")
            raise

    def cleanup(self):
        if os.path.exists(self.temp_dir_parent):
            print(f"Cleaning up temp directory: {self.temp_dir_parent}")
            shutil.rmtree(self.temp_dir_parent, ignore_errors=True)


# --- Updated FrameExtractor using FFmpeg Direct Output ---
class FrameExtractor:
    def extract_frames(self, video_path: str, target_fps: int) -> Tuple[List[Dict[str, Any]], float, float]:
        start_time_outer = time.time()
        print(f"Starting frame extraction using FFmpeg method for {video_path} at {target_fps} FPS")

        try:
            # Use FFmpeg probe for more reliable video info
            probe = ffmpeg.probe(video_path)
        except ffmpeg.Error as e:
            print(f"FFmpeg probe failed for {video_path}: {e.stderr.decode('utf8', errors='ignore')}")
            # Fallback to OpenCV probe if FFmpeg probe fails? Or just raise error?
            # Let's raise for now, as FFmpeg is required for extraction anyway.
            raise IOError(f"FFmpeg probe failed for {video_path}") from e

        video_stream_info = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if not video_stream_info:
            raise ValueError("No video stream found in file using FFmpeg probe.")
        
        # Get original FPS from probe data (preferred method)
        avg_frame_rate_str = video_stream_info.get('avg_frame_rate', '0/1')
        original_fps = 0.0
        if '/' in avg_frame_rate_str:
            try:
                num, den = map(int, avg_frame_rate_str.split('/'))
                original_fps = num / den if den != 0 else 0.0
            except ValueError:
                 print(f"Warning: Could not parse avg_frame_rate '{avg_frame_rate_str}'.")
                 original_fps = 0.0 # Will default below

        if original_fps <= 0: # Fallback if probe fails or provides invalid FPS
             # Try OpenCV probe as a secondary check, although FFmpeg probe is usually better
            try:
                cap_fallback = cv2.VideoCapture(video_path)
                if cap_fallback.isOpened():
                    original_fps = cap_fallback.get(cv2.CAP_PROP_FPS)
                    cap_fallback.release()
                    print(f"Warning: FFmpeg probe FPS was invalid ({avg_frame_rate_str}). Using OpenCV fallback FPS: {original_fps:.2f}")
            except Exception as cv_e:
                 print(f"Warning: OpenCV probe fallback also failed: {cv_e}")

        # Final default if all probes fail
        if original_fps <= 0:
            print(f"Warning: Original FPS could not be determined, defaulting to 30.0 FPS.")
            original_fps = 30.0
            
        # Calculate skip interval based on target FPS
        skip_interval = 1
        if target_fps > 0 and original_fps > target_fps:
            skip_interval = max(1, int(round(original_fps / target_fps))) # Ensure minimum skip is 1
        
        print(f"Original FPS: {original_fps:.2f}, Target FPS: {target_fps}, Skip Interval: {skip_interval}")

        frames_data = []
        temp_frame_dir = None
        
        try:
            # Create a temporary directory for FFmpeg output frames
            temp_frame_dir = tempfile.mkdtemp(prefix="ffmpeg_frames_")
            output_pattern = os.path.join(temp_frame_dir, "frame_%07d.jpg")

            # Build FFmpeg command
            # -vf select='not(mod(n,{skip_interval}))': Selects frames where frame number (n) modulo skip_interval is 0
            # -vcodec mjpeg: Output as motion JPEG (fast encoding)
            # -q:v 2: JPEG quality (2 is high quality, 1 is highest, 31 is lowest)
            # -vsync vfr: Use variable frame rate, important with select filter to match original timing
            # -y: Overwrite output if exists
            print(f"Running FFmpeg command to extract frames (filter: select='not(mod(n,{skip_interval}))', output: {output_pattern})")
            cmd = (
                ffmpeg
                .input(video_path)
                .output(output_pattern, vf=f"select='not(mod(n,{skip_interval}))'", vcodec='mjpeg', q=2, vsync='vfr')
                .overwrite_output() # -y flag
            )
            
            # Execute the command
            # capture_stdout/stderr is useful for debugging errors
            stdout, stderr = cmd.run(capture_stdout=True, capture_stderr=True)
            print("FFmpeg stdout:\n", stdout.decode('utf8', errors='ignore')[-500:]) # Print last 500 chars
            print("FFmpeg stderr:\n", stderr.decode('utf8', errors='ignore')[-500:]) # Print last 500 chars

            # Read the generated JPEG files
            output_frame_files = sorted(os.listdir(temp_frame_dir)) # Read directory and sort files by name
            print(f"Found {len(output_frame_files)} potential frame files in temp dir.")
            
            extracted_count = 0
            # Note: The frame_idx in the output files (frame_%07d.jpg) starts from 1.
            # We need to calculate the *original* frame index based on the skip interval.
            # If skip_interval is 1, frame_%07d.jpg corresponds to original frame (d-1).
            # If skip_interval is K, frame_%07d.jpg corresponds to original frame (d-1)*K.

            # Alternatively, and often simpler with select filter + vsync vfr:
            # The FFmpeg output frames are effectively sequential based on the selection.
            # The *timing* from the original video is preserved relative to the frames selected.
            # So, the 'frame_idx' we use later (for timestamp calculation) should probably just be the index in the *extracted* list.
            # Let's redefine 'frame_idx' in the output data to mean the index in the *sequence of extracted frames*,
            # and use the `timestamp_ms` field to represent the original video time. This is usually more useful.
            # We'll keep the original frame number logic below for clarity *if* you needed the original index,
            # but for timestamp calculation, the *sequential index* of the extracted frame is simpler with FFmpeg output.
            # Let's calculate timestamp based on the *sequential* index and the *target_fps*.

            # Revision: The original code used `frame_idx / original_fps * 1000` for timestamp.
            # This implies `frame_idx` should be the index in the *original* video.
            # The `select` filter with `not(mod(n,K))` selects frames 0, K, 2K, etc.
            # So, the d-th file outputted (starting d=1) corresponds to original frame index (d-1)*K.
            # Let's stick to that convention for `frame_idx` in the output data.

            for i, f_name in enumerate(output_frame_files):
                 # Filter for expected JPEG files
                if f_name.startswith("frame_") and f_name.endswith(".jpg"):
                    file_path = os.path.join(temp_frame_dir, f_name)
                    
                    # Basic check if file is likely a valid JPEG
                    if os.path.getsize(file_path) < 100: 
                         print(f"Warning: Skipping small file {f_name} (size: {os.path.getsize(file_path)})")
                         continue

                    try:
                         with open(file_path, 'rb') as f_jpg:
                            jpg_bytes = f_jpg.read()

                         # Calculate original frame index
                         # The file name is frame_0000001.jpg, frame_0000002.jpg, etc.
                         # The integer part is the sequential index from 1.
                         seq_idx_from_filename = int(f_name[6:13]) - 1 # e.g., "frame_0000001.jpg" -> 1 -> index 0
                         
                         # Original frame index = sequential index in extracted set * skip interval
                         # This assumes select filter correctly maps sequential output frames to original frame indices.
                         # Based on ffmpeg docs, select with vfr should work this way.
                         original_idx = seq_idx_from_filename * skip_interval

                         frames_data.append({
                             'frame_idx': original_idx, # This is the index in the *original* video
                             'frame_bytes_jpg': jpg_bytes,
                             'original_fps': original_fps # Use the FPS we detected/defaulted to
                         })
                         extracted_count += 1
                         if extracted_count % 100 == 0:
                             print(f"FFmpeg extracted {extracted_count} frames so far...")
                             
                    except Exception as read_err:
                         print(f"Error reading or processing temp file {f_name}: {read_err}")
                         # Continue to process other files
                         
        except ffmpeg.Error as e:
            print(f"FFMPEG execution error. Stderr: {e.stderr.decode('utf8', errors='ignore')}")
            # Re-raise the FFmpeg error
            raise
        except Exception as pipeline_err:
             print(f"Error during FFmpeg extraction pipeline: {pipeline_err}")
             # Re-raise other errors
             raise
        finally:
            # Clean up the temporary directory containing extracted frames
            if temp_frame_dir and os.path.exists(temp_frame_dir):
                print(f"Cleaning up temp frame directory: {temp_frame_dir}")
                shutil.rmtree(temp_frame_dir, ignore_errors=True) # ignore_errors for robustness
                
        extraction_time = time.time() - start_time_outer
        print(f"Finished FFmpeg extraction. Extracted {len(frames_data)} frames in {extraction_time:.2f} seconds.")
        
        # Ensure the original_fps used for timestamp calculation is returned
        return frames_data, original_fps, extraction_time

# --- Moondream Worker Class (Same as before) ---
@app.cls(gpu="a10g", timeout=180, image=moondream_image, max_containers=10, min_containers=0)
class MoondreamWorker:
    @modal.enter()
    def load_model(self):
        import torch
        self.device = "cuda"; self.dtype = torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, cache_dir=MODEL_CACHE_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, trust_remote_code=True, revision=REVISION,
            torch_dtype=self.dtype, cache_dir=MODEL_CACHE_PATH,
        ).to(self.device)
        self.model.eval()
        print("MoondreamWorker: Model loaded.")

    @modal.method()
    def describe_frame(self, frame_task: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        job_id_for_frame = frame_task['job_id_for_frame'] # Changed from video_id to avoid confusion
        frame_idx = frame_task['frame_idx'] # Keep the original frame index
        frame_bytes_jpg = frame_task['frame_bytes_jpg']
        original_fps = frame_task['original_fps'] # Use the FPS determined during extraction
        question = frame_task.get('question', "Describe this scene.")
        
        start_process_time = time.time()
        try:
            image_pil = Image.open(io.BytesIO(frame_bytes_jpg)).convert("RGB")
            with torch.no_grad():
                start_encode_time = time.time()
                image_embeds = self.model.encode_image(image_pil)
                encode_time_ms = (time.time() - start_encode_time) * 1000
                start_inference_time = time.time()
                answer = self.model.answer_question(
                    image_embeds=image_embeds, question=question,
                    tokenizer=self.tokenizer
                )
                inference_time_ms = (time.time() - start_inference_time) * 1000

            # Calculate timestamp using the original frame index and the determined FPS
            timestamp_ms = int(frame_idx / original_fps * 1000) if original_fps > 0 else 0
            
            total_worker_time_ms = (time.time() - start_process_time) * 1000
            
            return {
                'job_id_for_frame': job_id_for_frame, 'frame_idx': frame_idx, 'timestamp_ms': timestamp_ms,
                'description': answer,
                'processing_times_ms': {
                    'image_encode': round(encode_time_ms, 2),
                    'llm_inference': round(inference_time_ms, 2),
                    'total_worker_frame_time': round(total_worker_time_ms, 2),
                }, 'status': 'success'
            }
        except Exception as e:
            print(f"Error processing frame {frame_idx} for job {job_id_for_frame}: {e}")
            return {
                'job_id_for_frame': job_id_for_frame, 'frame_idx': frame_idx, 'status': 'error',
                'error_message': str(e)
            }

# --- Internal Processing Pipeline (Called by submit_video_job) ---
@app.function(timeout=3600) # Keeps a long timeout for the actual processing
def _process_video_pipeline_internal(job_id: str, video_url: str, target_fps: int, question_per_frame: str):
    print(f"INTERNAL: Starting processing for job_id: {job_id}, URL: {video_url}")
    overall_start_time = time.time()

    # Update job status to processing
    try:
        current_job_data = job_store.get(job_id, {}) # Default to empty dict if somehow missing
        current_job_data.update({
            "status": "processing",
            "processing_started_at": time.time(),
        })
        job_store[job_id] = current_job_data
    except Exception as e:
        print(f"INTERNAL: Error updating job {job_id} to processing: {e}")
        # Potentially mark job as failed immediately if this critical step fails
        job_store[job_id] = {**current_job_data, "status": "failed", "error_message": f"Failed to start processing: {e}"}
        return

    downloader = VideoDownloader()
    pipeline_output_data = {} # To store the structured results
    video_path = None # Initialize video_path
    try:
        video_path, download_time = downloader.download(video_url)
        print(f"Downloaded video in {download_time:.2f} seconds: {video_path}")
        
        extractor = FrameExtractor()
        frames_to_extract, original_fps, extraction_time = extractor.extract_frames(video_path, target_fps)

        if not frames_to_extract:
            # Even if some frames were extracted but were invalid (small size), the valid_frames_check in the extractor helps.
            # But if the list is empty, it's a failure.
            raise ValueError("No valid frames extracted from video.")

        num_frames = len(frames_to_extract)
        print(f"Extracted {num_frames} frames at {target_fps} fps. Preparing for parallel processing...")

        # Ensure original_fps is passed to workers for timestamp calculation
        tasks_for_workers = [{
            'job_id_for_frame': job_id, # Use the main job_id
            'frame_idx': fi['frame_idx'], # This is the original frame index as calculated by extractor
            'frame_bytes_jpg': fi['frame_bytes_jpg'],
            'original_fps': original_fps, # Pass the determined original FPS
            'question': question_per_frame
        } for fi in frames_to_extract]

        worker_instance = MoondreamWorker()
        all_frame_results = []
        map_start_time = time.time()
        
        # Process in batches to provide progress updates
        # Using a smaller batch size might be slightly better if individual frames are large
        batch_size = 100  # Adjust based on testing; 50 is often a good balance
        num_batches = (num_frames + batch_size - 1) // batch_size
        
        print(f"Processing {num_frames} frames in {num_batches} batches of size up to {batch_size}.")

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_frames)
            batch = tasks_for_workers[start_idx:end_idx]
            
            print(f"Processing batch {batch_idx+1}/{num_batches} ({start_idx} to {end_idx-1})...")
            # Use order_outputs=True to keep results in submitted order
            batch_results = list(worker_instance.describe_frame.map(batch, order_outputs=True))
            all_frame_results.extend(batch_results)
            
            # Update progress in job store
            progress_job_data = job_store.get(job_id, {})
            # Calculate progress based on number of tasks submitted vs results received
            processed_count_for_progress = len(all_frame_results) 
            total_count_for_progress = num_frames # Or len(tasks_for_workers)
            
            progress_percentage = 0
            if total_count_for_progress > 0:
                 progress_percentage = min(99, int((processed_count_for_progress / total_count_for_progress) * 100))

            progress_job_data.update({
                "progress": progress_percentage,
                "frames_processed": processed_count_for_progress,
                "total_frames_expected": total_count_for_progress # Use 'expected' or 'submitted'
            })
            job_store[job_id] = progress_job_data
            
            current_time = time.time()
            elapsed = current_time - map_start_time
            frames_per_second = processed_count_for_progress / elapsed if elapsed > 0 else 0
            
            # Simple ETA calculation - assumes constant rate
            remaining_frames = total_count_for_progress - processed_count_for_progress
            estimated_remaining = remaining_frames / frames_per_second if frames_per_second > 0 else float('inf')
            
            print(f"Progress: {progress_percentage}% - {processed_count_for_progress}/{total_count_for_progress} frames")
            print(f"Processing rate: {frames_per_second:.2f} frames/sec. Estimated remaining time: {estimated_remaining:.2f} seconds")
        
        map_duration = time.time() - map_start_time
        print(f"All frame descriptions processed in {map_duration:.2f} seconds")
        
        overall_duration = time.time() - overall_start_time
        pipeline_output_data = {
            "video_url": video_url,
            "target_fps_setting": target_fps,
            "actual_original_fps_detected": original_fps, # Store the FPS that was used
            "total_frames_submitted_to_workers": num_frames,
            "total_frames_results_received": len(all_frame_results),
            "timings_seconds": {
                "video_download": round(download_time, 2),
                "frame_extraction": round(extraction_time, 2),
                "distributed_frame_processing_map": round(map_duration, 2),
                "overall_pipeline_internal": round(overall_duration, 2),
            },
            "frame_results": all_frame_results
        }
        
        # Update job store with completed status and results
        final_job_data = job_store.get(job_id, {})
        final_job_data.update({
            "status": "completed",
            "results_payload": pipeline_output_data, # Store the detailed output here
            "completed_at": time.time(),
            "error_message": None
        })
        job_store[job_id] = final_job_data
        print(f"INTERNAL: Job {job_id} completed successfully.")

    except Exception as e:
        print(f"INTERNAL: Error in pipeline for job {job_id} ({video_url}): {e}")
        import traceback
        error_traceback = traceback.format_exc()
        # Update job store with failed status and error
        error_job_data = job_store.get(job_id, {})
        error_job_data.update({
            "status": "failed",
            "error_message": str(e),
            "traceback": error_traceback,
            "failed_at": time.time(),
            "results_payload": None # Clear any partial results
        })
        job_store[job_id] = error_job_data
        print(f"INTERNAL: Job {job_id} marked as failed.")
    finally:
        # Clean up the downloaded video file and its temp directory
        downloader.cleanup()

# --- Public Functions for Job Submission and Status (Same as before) ---

# Exposed as web endpoint for job submission
@app.function(min_containers=0)
@modal.asgi_app()
def api():
    from fastapi import FastAPI, HTTPException, Query, Path
    
    web_app = FastAPI(title="Moondream Video Processing API")
    
    @web_app.post("/api/submit")
    async def submit_job(video_url: str, target_fps: int = 1, question: str = "Describe this scene."):
        try:
            # Validate inputs
            if not video_url or not video_url.startswith(("http://", "https://")):
                raise HTTPException(status_code=400, detail="Invalid video URL. Must be a valid HTTP/HTTPS URL.")
            
            if target_fps <= 0 or target_fps > 30:
                raise HTTPException(status_code=400, detail="Invalid target_fps. Must be between 1 and 30.")
                
            # Submit the job directly
            job_id = f"job_{uuid.uuid4()}"
            initial_job_data = {
                "job_id": job_id,
                "video_url": video_url,
                "input_target_fps": target_fps,
                "input_question": question,
                "status": "submitted",
                "submitted_at": time.time(),
                "results_payload": None,
                "error_message": None,
                "traceback": None,
                "processing_started_at": None,
                "completed_at": None,
                "failed_at": None,
                "progress": 0, # Add initial progress
                "frames_processed": 0,
                "total_frames_expected": 0,
            }
            job_store[job_id] = initial_job_data
            
            # Spawn the internal processing pipeline
            _process_video_pipeline_internal.spawn(job_id, video_url, target_fps, question)
            
            return {"job_id": job_id, "status": "submitted", "message": "Job submitted for processing."}
        except Exception as e:
            # Log the exception for debugging
            import traceback
            print(f"Error submitting job: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}")
    
    @web_app.get("/api/status/{job_id}")
    async def get_job_status(job_id: str):
        try:
            if not job_id or not job_id.startswith("job_"):
                raise HTTPException(status_code=400, detail="Invalid job ID format.")
                
            # Get status directly from job store
            job_data = job_store.get(job_id)
            if job_data is None:
                raise HTTPException(status_code=404, detail="Job ID not found in store.")
            
            # Return a summary for ongoing/failed jobs, or full if completed
            # Exclude large payload unless completed
            summary_data = {k: v for k, v in job_data.items() if k != "results_payload"}
            if job_data.get("status") == "completed":
                 # Include payload if completed
                 summary_data["results_payload"] = job_data.get("results_payload")
                 
            return summary_data
        except HTTPException:
            raise
        except Exception as e:
            # Log the exception
            import traceback
            print(f"Error getting job status for {job_id}: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve job status: {str(e)}")
    
    @web_app.get("/api/jobs/list")
    async def list_jobs(limit: int = Query(default=10, ge=1, le=100), 
                         status_filter: str = Query(default=None)):
        try:
            # Get all job IDs from the job store
            # Note: Iterating job_store keys might be slow for very large stores.
            # For millions of jobs, a database or different store might be better.
            job_ids = list(job_store.keys())
            
            jobs_list_summary = []
            for job_id in job_ids:
                job_data = job_store.get(job_id)
                if not job_data:
                    continue # Should not happen if keys() returns valid keys
                    
                # Filter by status if specified
                if status_filter and job_data.get("status") != status_filter:
                    continue
                    
                # Create a summary without the full results payload or traceback
                job_summary = {
                    "job_id": job_data.get("job_id"),
                    "video_url": job_data.get("video_url"),
                    "status": job_data.get("status"),
                    "submitted_at": job_data.get("submitted_at"),
                    "completed_at": job_data.get("completed_at"),
                    "failed_at": job_data.get("failed_at"),
                    "progress": job_data.get("progress", 0),
                    "frames_processed": job_data.get("frames_processed", 0),
                    "total_frames_expected": job_data.get("total_frames_expected", 0),
                }
                jobs_list_summary.append(job_summary)
                
            # Sort by submission time (newest first)
            jobs_list_summary.sort(key=lambda x: x.get("submitted_at", 0), reverse=True)
            
            # Limit the results
            limited_jobs = jobs_list_summary[:limit]
                
            return {"jobs": limited_jobs, "total_count": len(jobs_list_summary)}
        except Exception as e:
            # Log the exception
            import traceback
            print(f"Error listing jobs: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")
    
    @web_app.get("/")
    async def homepage():
        return {
            "name": "Moondream Video Processing API",
            "description": "API for processing videos with Moondream2 vision-language model",
            "version": "1.0.0",
            "endpoints": [
                {"path": "/api/submit", "method": "POST", "description": "Submit a new video processing job"},
                {"path": "/api/status/{job_id}", "method": "GET", "description": "Check status of a specific job"},
                {"path": "/api/jobs/list", "method": "GET", "description": "List recent jobs"},
                {"path": "/", "method": "GET", "description": "This documentation page"}
            ],
            "model_info": {
                "model_id": MODEL_ID,
                "revision": REVISION
            }
        }
    
    return web_app


@app.local_entrypoint()
def main(
    video_url: str = "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4",
    target_fps: int = 1,
    question: str = "What is happening in this scene? Be concise.",
    output_json_file: str = "moondream_video_analysis_job.json"
):
    print(f"--- Moondream Video Analysis Job Submission (Modal) ---")
    print(f"Input Video URL: {video_url}")
    print(f"Target FPS: {target_fps}, Question: '{question}'")
    print(f"Full job data will be saved to: {output_json_file} upon completion.")
    print("----------------------------------------------------")

    # 1. Submit the job directly
    job_id = f"job_{uuid.uuid4()}"
    initial_job_data = {
        "job_id": job_id,
        "video_url": video_url,
        "input_target_fps": target_fps,
        "input_question": question,
        "status": "submitted",
        "submitted_at": time.time(),
        "results_payload": None, # Placeholder for results
        "error_message": None,
        "traceback": None,
        "processing_started_at": None,
        "completed_at": None,
        "failed_at": None,
        "progress": 0, # Initial progress fields
        "frames_processed": 0,
        "total_frames_expected": 0,
    }
    job_store[job_id] = initial_job_data
    print(f"Job submitted successfully! Job ID: {job_id}")

    # Spawn the processing pipeline
    _process_video_pipeline_internal.spawn(job_id, video_url, target_fps, question)
    
    # 2. Poll for status
    print(f"\nPolling for job status (Job ID: {job_id})... Press Ctrl+C to stop early.")
    final_job_data = None
    try:
        while True:
            time.sleep(5) # Poll every 5 seconds for slightly quicker updates
            job_data = job_store.get(job_id)
            
            if not job_data:
                print(f"  [{time.strftime('%H:%M:%S')}] Error: No status response for job {job_id}. It might have failed early or been purged.")
                # Assume failure if status is missing after submission
                final_job_data = {"job_id": job_id, "status": "not_found_or_failed_early", "error_message": "Job data not found in store."}
                break

            current_status = job_data.get("status", "unknown")
            progress = job_data.get("progress", 0)
            frames_processed = job_data.get("frames_processed", 0)
            total_frames = job_data.get("total_frames_expected", 0) # Use the expected count from job data
            
            status_line = f"  [{time.strftime('%H:%M:%S')}] Job {job_id} status: {current_status}"
            if current_status == "processing":
                 status_line += f" ({progress}% - {frames_processed}/{total_frames} frames)"
                 
            print(status_line)

            if current_status in ["completed", "failed"]: # Check for completion or failure
                final_job_data = job_data
                break
            elif current_status == "submitted" and (time.time() - job_data.get("submitted_at", time.time())) > 60:
                 # If still 'submitted' after a minute, something might be stuck
                 print(f"  [{time.strftime('%H:%M:%S')}] Warning: Job {job_id} still in 'submitted' state for over 60 seconds. Potential issue launching.")
                 # Decide if you want to break or continue polling. Let's continue polling for now.
                 pass # Keep polling
                 
            # Add a check for 'not_found' status in the store result itself, though get() returning None handles the common case
            if current_status == "not_found": # Should be handled by job_data is None check above
                 print(f"  [{time.strftime('%H:%M:%S')}] Job {job_id} status: {current_status}. Ending poll.")
                 final_job_data = job_data
                 break
                 
    except KeyboardInterrupt:
        print("\nPolling interrupted by user. Fetching final status...")
        final_job_data = job_store.get(job_id) # Get one last status
    except Exception as e:
         print(f"\nAn unexpected error occurred during polling: {e}")
         # Try to get the last known status
         final_job_data = job_store.get(job_id)

    # 3. Process and save final results
    if final_job_data:
        print(f"\n--- Final Job Status for {job_id} ---")
        status = final_job_data.get("status")
        
        # Determine output file name
        output_filename_to_save = output_json_file
        if status == "failed":
             output_filename_to_save = output_json_file.replace(".json", "_error.json")
             print("Status: FAILED")
             print(f"Error Message: {final_job_data.get('error_message', 'N/A')}")
             if final_job_data.get('traceback'):
                 print(f"Traceback (partial):\n{final_job_data['traceback'][:1000]}...") # Print partial traceback
        elif status == "completed":
            print("Status: COMPLETED")
            results_payload = final_job_data.get("results_payload")
            if not results_payload:
                 print("Job completed, but no 'results_payload' found in the job data.")
        elif status == "not_found" or status == "not_found_or_failed_early":
             print(f"Status: {status.upper()}")
             print(f"Job ID {job_id} was not found in the system or failed very early.")
        else:
            print(f"Status: {status.upper() if status else 'UNKNOWN'}")
            
        # Save the final job data (whether completed or failed)
        try:
            with open(output_filename_to_save, 'w') as f:
                # Exclude potentially very large byte data or traceback from default print, but save to file
                data_to_save = final_job_data.copy()
                # Example: Optionally remove frame bytes from results before saving if file size is a concern,
                # but for analysis, keeping them might be useful if needed later (though it's redundant after processing)
                # For now, let's save the full results_payload if it exists and job is completed.
                # If job is failed, payload is usually None anyway.
                
                json.dump(data_to_save, f, indent=2) 
            print(f"Full job data saved to: {output_filename_to_save}")
        except IOError as e:
            print(f"Error saving job data to JSON file {output_filename_to_save}: {e}")
        except Exception as e:
             print(f"An unexpected error occurred while trying to save job data: {e}")


        # Print sample descriptions only if completed and results exist
        if status == "completed" and results_payload and results_payload.get('frame_results'):
            print("\n--- Sample Frame Descriptions ---")
            sample_count = 3
            for i, res in enumerate(results_payload['frame_results'][:sample_count]): # Print first N
                if res.get('status') == 'success':
                    # Include original frame index and timestamp
                    print(f"  Frame {res.get('frame_idx', 'N/A')} (Ts: {res.get('timestamp_ms', 'N/A')}ms): {res.get('description', 'N/A')}")
                else:
                    print(f"  Frame {res.get('frame_idx', 'N/A')}: Error - {res.get('error_message', 'Unknown')}")
            if len(results_payload['frame_results']) > sample_count:
                print(f"  ... and {len(results_payload['frame_results']) - sample_count} more frames processed.")

    else:
        print("\nCould not retrieve final job status after polling.")
    print("---------------------------------------")