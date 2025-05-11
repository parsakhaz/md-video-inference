#!/usr/bin/env python3
"""
Benchmark utility for determining optimal model concurrency on a GPU.
This simulates loading multiple instances of the Moondream model to determine the sweet spot.
"""

import os
import time
import torch
import tempfile
import requests
import cv2
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import io
import logging
import gc
import json
from typing import List, Dict, Any, Tuple, Optional

class BenchmarkUtility:
    """Utility for benchmarking model concurrency on a GPU"""
    
    MODEL_ID = "vikhyatk/moondream2"
    REVISION = "2024-05-20"
    
    def __init__(self, debug_log: bool = False):
        """Initialize the benchmark utility"""
        # Set up logging
        log_level = logging.DEBUG if debug_log else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('benchmark_utility')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16
        
        if self.device == "cpu":
            self.logger.warning("Running on CPU, benchmarking will be slow and may not reflect actual GPU performance")
        else:
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"CUDA Version: {torch.version.cuda}")
            self.logger.info(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    def _download_video(self, video_url: str) -> str:
        """Download a video from URL to local temp file"""
        self.logger.info(f"Downloading video from {video_url}")
        
        temp_dir = tempfile.mkdtemp(prefix="benchmark_downloads_")
        try:
            fname = video_url.split("/")[-1]
            if not fname or "." not in fname:
                fname = "benchmark_video.mp4"
                
            video_filename = os.path.join(temp_dir, fname)
            with requests.get(video_url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(video_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
            self.logger.info(f"Video downloaded to {video_filename}")
            return video_filename
            
        except Exception as e:
            self.logger.error(f"Error downloading video: {str(e)}")
            raise
    
    def _extract_frames(self, video_path: str, fps: int) -> List[bytes]:
        """Extract frames from video at specified FPS"""
        self.logger.info(f"Extracting frames from {video_path} at {fps} FPS")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
            
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps == 0:
            original_fps = 30.0
            
        self.logger.info(f"Original video FPS: {original_fps}")
        skip_interval = int(round(original_fps / fps)) if fps > 0 and original_fps > fps else 1
        
        frames_bytes = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % skip_interval == 0:
                is_success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if is_success:
                    frames_bytes.append(buffer.tobytes())
                    
            frame_idx += 1
            
        cap.release()
        self.logger.info(f"Extracted {len(frames_bytes)} frames")
        return frames_bytes
    
    def _load_single_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load a single instance of the model and tokenizer"""
        self.logger.info(f"Loading model {self.MODEL_ID} (revision {self.REVISION})")
        
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID, revision=self.REVISION)
        model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=True,
            revision=self.REVISION,
            torch_dtype=self.dtype,
        ).to(self.device)
        model.eval()
        
        return model, tokenizer
    
    def _process_frame_with_model(
        self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, frame_bytes: bytes, question: str = "Describe this scene."
    ) -> Tuple[str, Dict[str, float]]:
        """Process a single frame with the model and measure performance"""
        start_total_time = time.time()
        
        # Load image
        image_pil = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
        
        # Process with model
        with torch.no_grad():
            # Encode image
            start_encode_time = time.time()
            image_embeds = model.encode_image(image_pil)
            encode_time = time.time() - start_encode_time
            
            # Generate answer
            start_inference_time = time.time()
            answer = model.answer_question(
                image_embeds=image_embeds,
                question=question,
                tokenizer=tokenizer,
                max_new_tokens=128
            )
            inference_time = time.time() - start_inference_time
        
        total_time = time.time() - start_total_time
        
        # Return timing metrics
        timing = {
            "encode_time_ms": encode_time * 1000,
            "inference_time_ms": inference_time * 1000,
            "total_time_ms": total_time * 1000
        }
        
        return answer, timing
    
    def _clear_gpu_memory(self):
        """Clear GPU memory between benchmark runs"""
        if self.device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
            self.logger.info(f"Cleared GPU memory. Available: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    def benchmark_concurrency(self, num_models: int, frame_bytes: List[bytes]) -> Dict[str, Any]:
        """Benchmark a specific concurrency level"""
        self.logger.info(f"Benchmarking with {num_models} concurrent models")
        
        # Clear memory from previous runs
        self._clear_gpu_memory()
        
        results = {
            "num_models": num_models,
            "memory_before_mb": torch.cuda.memory_allocated(0) / (1024**2) if self.device == "cuda" else 0,
            "frames_processed": 0,
            "failed_frames": 0,
            "timings_ms": [],
            "total_time_sec": 0,
            "throughput_fps": 0,
            "error": None
        }
        
        try:
            # Load multiple model instances
            models_and_tokenizers = []
            for i in range(num_models):
                self.logger.info(f"Loading model instance {i+1}/{num_models}")
                model, tokenizer = self._load_single_model()
                models_and_tokenizers.append((model, tokenizer))
            
            results["memory_after_loading_mb"] = torch.cuda.memory_allocated(0) / (1024**2) if self.device == "cuda" else 0
            
            # Process frames round-robin with different model instances
            start_time = time.time()
            
            # Limit to 100 frames for consistency and to avoid excessive benchmark time
            frames_to_process = frame_bytes[:min(100, len(frame_bytes))]
            
            for i, frame in enumerate(frames_to_process):
                model_idx = i % num_models
                model, tokenizer = models_and_tokenizers[model_idx]
                
                try:
                    answer, timing = self._process_frame_with_model(model, tokenizer, frame)
                    results["frames_processed"] += 1
                    results["timings_ms"].append(timing)
                    
                    # Log every 10 frames
                    if i % 10 == 0 or i == len(frames_to_process) - 1:
                        self.logger.info(f"Processed frame {i+1}/{len(frames_to_process)} with model {model_idx+1}")
                        
                except Exception as e:
                    self.logger.error(f"Failed to process frame {i} with model {model_idx}: {str(e)}")
                    results["failed_frames"] += 1
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Calculate metrics
            results["total_time_sec"] = total_time
            results["throughput_fps"] = results["frames_processed"] / total_time if total_time > 0 else 0
            
            # Calculate average timing metrics
            if results["timings_ms"]:
                avg_timings = {
                    "avg_encode_time_ms": sum(t["encode_time_ms"] for t in results["timings_ms"]) / len(results["timings_ms"]),
                    "avg_inference_time_ms": sum(t["inference_time_ms"] for t in results["timings_ms"]) / len(results["timings_ms"]),
                    "avg_total_time_ms": sum(t["total_time_ms"] for t in results["timings_ms"]) / len(results["timings_ms"])
                }
                results["avg_timings_ms"] = avg_timings
            
            # Clean up models to free memory
            for model, _ in models_and_tokenizers:
                del model
            models_and_tokenizers.clear()
            self._clear_gpu_memory()
            
        except Exception as e:
            self.logger.error(f"Error during benchmark with {num_models} models: {str(e)}")
            results["error"] = str(e)
            
            # Clean up in case of error
            try:
                for model, _ in models_and_tokenizers:
                    del model
                models_and_tokenizers.clear()
                self._clear_gpu_memory()
            except:
                pass
        
        return results
    
    def run_full_benchmark(self, video_url: str, max_models: int = 10, fps: int = 3):
        """Run a full benchmark across different numbers of concurrent models"""
        self.logger.info(f"Starting full benchmark with max_models={max_models}, fps={fps}")
        
        try:
            # Download video
            video_path = self._download_video(video_url)
            
            # Extract frames
            frame_bytes = self._extract_frames(video_path, fps)
            
            if not frame_bytes:
                raise ValueError("No frames extracted from video")
            
            # Run benchmarks for different concurrency levels
            benchmark_results = []
            
            # Test with increasing number of model instances
            for num_models in range(1, max_models + 1):
                result = self.benchmark_concurrency(num_models, frame_bytes)
                benchmark_results.append(result)
                
                self.logger.info(f"Results for {num_models} models:")
                self.logger.info(f"  Throughput: {result['throughput_fps']:.2f} frames/second")
                self.logger.info(f"  Avg processing time: {result.get('avg_timings_ms', {}).get('avg_total_time_ms', 0):.2f} ms")
                self.logger.info(f"  Memory usage: {result['memory_after_loading_mb']:.2f} MB")
                
                # If we encounter an error, stop increasing the number of models
                if result["error"]:
                    self.logger.warning(f"Stopping benchmark due to error at {num_models} models")
                    break
            
            # Find optimal concurrency
            valid_results = [r for r in benchmark_results if not r["error"] and r["throughput_fps"] > 0]
            if valid_results:
                # Sort by throughput (descending)
                optimal_by_throughput = sorted(valid_results, key=lambda x: x["throughput_fps"], reverse=True)[0]
                
                self.logger.info("\n" + "="*50)
                self.logger.info("BENCHMARK RESULTS SUMMARY")
                self.logger.info("="*50)
                self.logger.info(f"Optimal concurrency: {optimal_by_throughput['num_models']} models")
                self.logger.info(f"Best throughput: {optimal_by_throughput['throughput_fps']:.2f} frames/second")
                
                # Save results to JSON file
                result_file = "benchmark_results.json"
                with open(result_file, "w") as f:
                    json.dump(benchmark_results, f, indent=2)
                self.logger.info(f"Detailed results saved to {result_file}")
                
            else:
                self.logger.error("No valid benchmark results were obtained")
                
        except Exception as e:
            self.logger.error(f"Error in benchmark: {str(e)}")
            raise 