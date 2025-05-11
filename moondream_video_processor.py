import modal
import os
import time
import tempfile
import shutil
import requests
import ffmpeg # ffmpeg-python
import cv2
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any, Tuple
import io
import json
import uuid # For generating unique job IDs

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
    .pip_install(
        "transformers==4.39.3", "torch==2.1.2", "torchaudio", "torchvision",
        "Pillow==10.0.0", "opencv-python-headless==4.8.0.76", "numpy==1.24.0",
        "requests==2.28.0", "ffmpeg-python==0.2.0", "accelerate", "fastapi",
    )
    .apt_install("ffmpeg")
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


# --- Video Processing Utilities (Same as before) ---
class VideoDownloader:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="video_downloads_")
    def download(self, video_url: str) -> Tuple[str, float]:
        start_time = time.time()
        try:
            fname = video_url.split("/")[-1]
            if not fname or "." not in fname: fname = "downloaded_video.mp4"
            video_filename = os.path.join(self.temp_dir, fname)
            with requests.get(video_url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(video_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
            download_time = time.time() - start_time
            return video_filename, download_time
        except Exception as e:
            self.cleanup()
            raise
    def cleanup(self):
        if os.path.exists(self.temp_dir): shutil.rmtree(self.temp_dir)

class FrameExtractor:
    def extract_frames(self, video_path: str, target_fps: int) -> Tuple[List[Dict[str, Any]], float, float]:
        start_time = time.time()
        if not os.path.exists(video_path): raise FileNotFoundError(f"Video not found: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise IOError(f"Cannot open video: {video_path}")
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps if original_fps > 0 else 0
        
        # Log some stats about the video
        print(f"Video has {total_frames} frames at {original_fps} fps (duration: {duration:.2f} seconds)")
        
        if original_fps == 0: original_fps = 30.0
        skip_interval = int(round(original_fps / target_fps)) if target_fps > 0 and original_fps > target_fps else 1
        
        # Estimate total frames to be extracted
        estimated_frames = total_frames // skip_interval
        print(f"Will extract approximately {estimated_frames} frames (1 every {skip_interval} frames)")
        
        frames_data = []
        current_frame_idx, extracted_count = 0, 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if current_frame_idx % skip_interval == 0:
                is_success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if is_success:
                    frames_data.append({
                        'frame_idx': current_frame_idx,
                        'frame_bytes_jpg': buffer.tobytes(),
                        'original_fps': original_fps
                    })
                    extracted_count += 1
                    if extracted_count % 100 == 0:
                        print(f"Extracted {extracted_count} frames so far...")
            current_frame_idx += 1
        cap.release()
        extraction_time = time.time() - start_time
        print(f"Extracted {extracted_count} frames in {extraction_time:.2f} seconds")
        return frames_data, original_fps, extraction_time

# --- Moondream Worker Class (Same as before) ---
@app.cls(gpu="a10g", timeout=180, image=moondream_image, max_containers=30, min_containers=1)
class MoondreamWorker:
    @modal.enter()
    def load_model(self):
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
        job_id_for_frame = frame_task['job_id_for_frame'] # Changed from video_id to avoid confusion
        frame_idx, frame_bytes_jpg = frame_task['frame_idx'], frame_task['frame_bytes_jpg']
        original_fps, question = frame_task['original_fps'], frame_task.get('question', "Describe this scene.")
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
                    tokenizer=self.tokenizer, max_new_tokens=128
                )
                inference_time_ms = (time.time() - start_inference_time) * 1000
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
    try:
        video_path, download_time = downloader.download(video_url)
        print(f"Downloaded video in {download_time:.2f} seconds: {video_path}")
        
        extractor = FrameExtractor()
        frames_to_extract, original_fps, extraction_time = extractor.extract_frames(video_path, target_fps)

        if not frames_to_extract:
            raise ValueError("No frames extracted from video.")

        num_frames = len(frames_to_extract)
        print(f"Extracted {num_frames} frames at {target_fps} fps. Preparing for parallel processing...")

        tasks_for_workers = [{
            'job_id_for_frame': job_id, # Use the main job_id
            'frame_idx': fi['frame_idx'], 'frame_bytes_jpg': fi['frame_bytes_jpg'],
            'original_fps': fi['original_fps'], 'question': question_per_frame
        } for fi in frames_to_extract]

        worker_instance = MoondreamWorker()
        all_frame_results = []
        map_start_time = time.time()
        
        # Process in batches to provide progress updates
        batch_size = 50  # Adjust based on expected frame count
        num_batches = (num_frames + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_frames)
            batch = tasks_for_workers[start_idx:end_idx]
            
            print(f"Processing batch {batch_idx+1}/{num_batches} ({start_idx} to {end_idx-1})...")
            batch_results = list(worker_instance.describe_frame.map(batch, order_outputs=True))
            all_frame_results.extend(batch_results)
            
            # Update progress in job store
            progress_job_data = job_store.get(job_id, {})
            progress_percentage = min(99, int((len(all_frame_results) / num_frames) * 100))
            progress_job_data.update({
                "progress": progress_percentage,
                "frames_processed": len(all_frame_results),
                "total_frames": num_frames
            })
            job_store[job_id] = progress_job_data
            
            current_time = time.time()
            elapsed = current_time - map_start_time
            frames_per_second = len(all_frame_results) / elapsed if elapsed > 0 else 0
            estimated_remaining = (num_frames - len(all_frame_results)) / frames_per_second if frames_per_second > 0 else 0
            
            print(f"Progress: {progress_percentage}% - {len(all_frame_results)}/{num_frames} frames")
            print(f"Processing rate: {frames_per_second:.2f} frames/sec")
            print(f"Estimated remaining time: {estimated_remaining:.2f} seconds")
        
        map_duration = time.time() - map_start_time
        print(f"All frames processed in {map_duration:.2f} seconds")
        
        overall_duration = time.time() - overall_start_time
        pipeline_output_data = {
            "video_url": video_url,
            "target_fps_setting": target_fps,
            "actual_original_fps": original_fps,
            "total_frames_submitted_to_workers": len(tasks_for_workers),
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
        downloader.cleanup()

# --- Public Functions for Job Submission and Status ---

# Exposed as web endpoint for job submission
@app.function(min_containers=1)
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
            }
            job_store[job_id] = initial_job_data
            
            # Spawn the internal processing pipeline
            _process_video_pipeline_internal.spawn(job_id, video_url, target_fps, question)
            
            return {"job_id": job_id, "status": "submitted", "message": "Job submitted for processing."}
        except Exception as e:
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
            summary_data = {
                key: value for key, value in job_data.items()
                if key != "results_payload" or job_data.get("status") == "completed"
            }
            if job_data.get("status") != "completed" and "results_payload" in job_data:
                summary_data["results_preview"] = "Results available upon completion."
                
            return summary_data
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to retrieve job status: {str(e)}")
    
    @web_app.get("/api/jobs/list")
    async def list_jobs(limit: int = Query(default=10, ge=1, le=100), 
                         status_filter: str = Query(default=None)):
        try:
            # Get all job IDs from the job store
            job_ids = list(job_store.keys())
            
            # Sort jobs by submission time (newest first)
            jobs_with_data = []
            for job_id in job_ids:
                job_data = job_store.get(job_id)
                if not job_data:
                    continue
                    
                # Filter by status if specified
                if status_filter and job_data.get("status") != status_filter:
                    continue
                    
                # Create a summary without the full results payload
                job_summary = {
                    "job_id": job_data.get("job_id"),
                    "video_url": job_data.get("video_url"),
                    "status": job_data.get("status"),
                    "submitted_at": job_data.get("submitted_at"),
                    "completed_at": job_data.get("completed_at"),
                    "failed_at": job_data.get("failed_at"),
                }
                jobs_with_data.append(job_summary)
                
            # Sort by submission time (newest first)
            jobs_with_data.sort(key=lambda x: x.get("submitted_at", 0), reverse=True)
            
            # Limit the results
            jobs_with_data = jobs_with_data[:limit]
                
            return {"jobs": jobs_with_data, "total_count": len(jobs_with_data)}
        except Exception as e:
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
            time.sleep(10) # Poll every 10 seconds
            job_data = job_store.get(job_id)
            
            if not job_data:
                print(f"  [{time.strftime('%H:%M:%S')}] Error: No status response for job {job_id}.")
                continue

            current_status = job_data.get("status", "unknown")
            progress = job_data.get("progress", 0)
            if progress > 0 and progress < 100 and current_status == "processing":
                frames_processed = job_data.get("frames_processed", 0)
                total_frames = job_data.get("total_frames", 0)
                print(f"  [{time.strftime('%H:%M:%S')}] Job {job_id} status: {current_status} ({progress}% - {frames_processed}/{total_frames} frames)")
            else:
                print(f"  [{time.strftime('%H:%M:%S')}] Job {job_id} status: {current_status}")

            if current_status in ["completed", "failed", "not_found"]:
                final_job_data = job_data
                break
    except KeyboardInterrupt:
        print("\nPolling interrupted by user. Fetching final status...")
        final_job_data = job_store.get(job_id) # Get one last status
    
    # 3. Process and save final results
    if final_job_data:
        print(f"\n--- Final Job Status for {job_id} ---")
        status = final_job_data.get("status")
        if status == "completed":
            print("Status: COMPLETED")
            results_payload = final_job_data.get("results_payload")
            if results_payload:
                try:
                    with open(output_json_file, 'w') as f:
                        json.dump(final_job_data, f, indent=2) # Save the whole job_data object
                    print(f"Full job data (including results) saved to: {output_json_file}")
                except IOError as e:
                    print(f"Error saving results to JSON: {e}")

                if results_payload.get('frame_results'):
                    print("\n--- Sample Frame Descriptions ---")
                    for i, res in enumerate(results_payload['frame_results'][:3]): # Print first 3
                        if res.get('status') == 'success':
                            print(f"  Frame {res['frame_idx']} (Ts: {res['timestamp_ms']}ms): {res['description']}")
                        else:
                            print(f"  Frame {res['frame_idx']}: Error - {res.get('error_message', 'Unknown')}")
                    if len(results_payload['frame_results']) > 3:
                        print(f"  ... and {len(results_payload['frame_results']) - 3} more frames processed.")
            else:
                print("Job completed, but no 'results_payload' found in the job data.")

        elif status == "failed":
            print("Status: FAILED")
            print(f"Error Message: {final_job_data.get('error_message', 'N/A')}")
            if final_job_data.get('traceback'):
                print(f"Traceback:\n{final_job_data['traceback'][:500]}...") # Print partial traceback
            try: # Save error details
                with open(output_json_file.replace(".json", "_error.json"), 'w') as f:
                    json.dump(final_job_data, f, indent=2)
                print(f"Error details saved to: {output_json_file.replace('.json', '_error.json')}")
            except IOError: pass

        elif status == "not_found":
            print("Status: NOT FOUND")
            print(f"The job ID {job_id} was not found in the system.")
        else:
            print(f"Status: {status.upper() if status else 'UNKNOWN'}")
            print(f"Full job data: {json.dumps(final_job_data, indent=2)}")
    else:
        print("\nCould not retrieve final job status after polling.")
    print("---------------------------------------") 