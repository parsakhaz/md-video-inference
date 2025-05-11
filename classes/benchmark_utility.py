import os
import logging
import datetime
import json
import torch
import gc
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
from .predictor import Predictor
from .video_analyzer import VideoAnalyzer

class BenchmarkUtility:
    """Utility class for benchmarking model performance and finding optimal model count"""
    
    def __init__(self, debug_log: bool = False):
        """
        Initialize the benchmark utility
        
        Args:
            debug_log: Whether to print detailed timing logs
        """
        self.debug_log = debug_log
        self.logger = self._setup_logger()
        self.gpu_info = self._get_gpu_info()
        self.results: List[Dict[str, Any]] = []
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging with file handler"""
        logger = logging.getLogger("benchmark_utility")
        
        if not logger.handlers:
            # Create a timestamp for log files
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create logs directory if it doesn't exist
            os.makedirs("logs", exist_ok=True)
            
            # Create a file handler for benchmark logs
            benchmark_log_file = f"logs/benchmark_{timestamp}.log"
            file_handler = logging.FileHandler(benchmark_log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)
            
            if self.debug_log:
                logger.setLevel(logging.DEBUG)
            else:
                logger.setLevel(logging.INFO)
                
            logger.info(f"Benchmark logging enabled. Logs will be saved to {benchmark_log_file}")
        
        return logger
        
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get information about the GPU"""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA is not available. Running on CPU will be very slow!")
            return {
                "cuda_available": False,
                "device_count": 0,
                "device_name": "CPU",
                "total_memory": 0
            }
        
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        
        self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
        self.logger.info(f"CUDA device count: {device_count}")
        self.logger.info(f"CUDA device: {device_name}")
        self.logger.info(f"Total GPU memory: {total_memory:.2f} GB")
        
        return {
            "cuda_available": True,
            "device_count": device_count,
            "device_name": device_name,
            "total_memory": total_memory
        }
        
    def run_benchmark(self, video_url: str, num_models: int, fps: int = 3) -> Optional[Dict[str, Any]]:
        """
        Run a benchmark with the specified number of models
        
        Args:
            video_url: URL of the video to analyze
            num_models: Number of model instances to test
            fps: Frames per second to analyze
            
        Returns:
            Dictionary of timing metrics if successful, None if failed
        """
        self.logger.info(f"Running benchmark with {num_models} models...")
        
        # Clean up memory before loading models
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            # Initialize predictor and analyzer
            predictor = Predictor()
            predictor.setup(num_models=num_models)
            
            # Create video analyzer
            analyzer = VideoAnalyzer(predictor.models, predictor.tokenizer_moondream, self.debug_log)
            
            # Run analysis
            results, timing_metrics = analyzer.analyze_video(
                video_url,
                frames_per_second=fps,
                use_queue=True  # Always use queue for consistency
            )
            
            # Add number of models to metrics
            timing_metrics["num_models"] = num_models
            
            # Log the results
            self.logger.info(f"Benchmark with {num_models} models completed successfully")
            self.logger.info(f"Processing time: {timing_metrics.get('Processing time (seconds)', 'N/A')} seconds")
            self.logger.info(f"Average time per frame: {timing_metrics.get('Average time per frame (ms)', 'N/A')} ms")
            
            return timing_metrics
            
        except Exception as e:
            self.logger.error(f"Error during benchmark with {num_models} models: {str(e)}")
            return None
            
        finally:
            # Clean up memory
            if 'predictor' in locals():
                del predictor
            if 'analyzer' in locals():
                del analyzer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    def find_sweet_spot(self) -> Optional[Dict[str, Any]]:
        """Find the sweet spot based on performance metrics"""
        if not self.results:
            return None
        
        try:
            # Extract metrics
            num_models = [r["num_models"] for r in self.results]
            
            # Get processing times, filtering out any None or missing values
            processing_times = []
            valid_indices = []
            
            for i, r in enumerate(self.results):
                if "metrics" in r and "Processing time" in r["metrics"] and r["metrics"]["Processing time"] is not None:
                    processing_times.append(r["metrics"]["Processing time"])
                    valid_indices.append(i)
            
            if not processing_times:
                self.logger.warning("No valid processing times found in results")
                return None
                
            # Find the minimum processing time
            min_processing_time = min(processing_times)
            min_processing_time_idx = valid_indices[processing_times.index(min_processing_time)]
            sweet_spot_models = num_models[min_processing_time_idx]
            
            # Get frame times, filtering out any None or missing values
            frame_times = []
            valid_frame_indices = []
            
            for i, r in enumerate(self.results):
                if "avg_time_per_frame_ms" in r and r["avg_time_per_frame_ms"] is not None:
                    frame_times.append(r["avg_time_per_frame_ms"])
                    valid_frame_indices.append(i)
            
            if frame_times:
                # Find the minimum average frame time
                min_frame_time = min(frame_times)
                min_frame_time_idx = valid_frame_indices[frame_times.index(min_frame_time)]
                sweet_spot_frame_models = num_models[min_frame_time_idx]
            else:
                min_frame_time = None
                sweet_spot_frame_models = None
            
            # If they're different, use the one with fewer models
            if sweet_spot_frame_models is not None:
                sweet_spot = min(sweet_spot_models, sweet_spot_frame_models)
            else:
                sweet_spot = sweet_spot_models
            
            return {
                "sweet_spot": sweet_spot,
                "min_processing_time": min_processing_time,
                "min_frame_time": min_frame_time,
                "sweet_spot_processing_time": sweet_spot_models,
                "sweet_spot_frame_time": sweet_spot_frame_models
            }
        except Exception as e:
            self.logger.error(f"Error finding sweet spot: {str(e)}")
            return None
        
    def plot_results(self, output_file: str) -> None:
        """
        Generate plots of the benchmark results
        
        Args:
            output_file: Path to save the plot file
        """
        if not self.results:
            self.logger.warning("No results to plot")
            return
        
        try:
            # Extract metrics, filtering out any None or missing values
            num_models = []
            processing_times = []
            frame_times = []
            
            for r in self.results:
                if "num_models" in r and "metrics" in r and "Processing time" in r["metrics"] and r["metrics"]["Processing time"] is not None:
                    num_models.append(r["num_models"])
                    processing_times.append(r["metrics"]["Processing time"])
                    
                    if "avg_time_per_frame_ms" in r and r["avg_time_per_frame_ms"] is not None:
                        frame_times.append(r["avg_time_per_frame_ms"])
                    else:
                        frame_times.append(None)
            
            if not num_models or not processing_times:
                self.logger.warning("No valid data to plot")
                return
                
            # Create figure with subplots
            fig, axs = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot processing time
            axs[0].plot(num_models, processing_times, 'o-', color='blue')
            axs[0].set_xlabel('Number of Models')
            axs[0].set_ylabel('Processing Time (seconds)')
            axs[0].set_title('Processing Time vs. Number of Models')
            axs[0].grid(True)
            
            # Plot average frame time if available
            if any(t is not None for t in frame_times):
                # Filter out None values
                valid_frame_indices = [i for i, t in enumerate(frame_times) if t is not None]
                valid_num_models = [num_models[i] for i in valid_frame_indices]
                valid_frame_times = [frame_times[i] for i in valid_frame_indices]
                
                axs[1].plot(valid_num_models, valid_frame_times, 'o-', color='green')
                axs[1].set_xlabel('Number of Models')
                axs[1].set_ylabel('Average Time per Frame (ms)')
                axs[1].set_title('Average Frame Time vs. Number of Models')
                axs[1].grid(True)
            else:
                axs[1].text(0.5, 0.5, 'No frame time data available', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=axs[1].transAxes)
                axs[1].set_title('Average Frame Time vs. Number of Models')
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(output_file)
            self.logger.info(f"Plots saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error generating plots: {str(e)}")
        
    def run_full_benchmark(self, video_url: str, max_models: int = 10, fps: int = 3) -> Dict[str, Any]:
        """
        Run a full benchmark test suite
        
        Args:
            video_url: URL of the video to analyze
            max_models: Maximum number of models to test
            fps: Frames per second to analyze
            
        Returns:
            Dictionary containing benchmark results and analysis
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting Full Benchmark Suite")
        self.logger.info("=" * 80)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Run benchmarks for each model count
            for num_models in range(1, max_models + 1):
                self.logger.info(f"\nTesting with {num_models} models...")
                
                timing_metrics = self.run_benchmark(video_url, num_models, fps)
                
                if timing_metrics:
                    self.results.append(timing_metrics)
                    
                    # Check if we're running out of memory
                    if torch.cuda.is_available():
                        memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
                        memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)  # GB
                        self.logger.info(f"GPU memory allocated: {memory_allocated:.2f} GB")
                        self.logger.info(f"GPU memory reserved: {memory_reserved:.2f} GB")
                        
                        # If we're using more than 95% of available memory, stop
                        if memory_reserved > self.gpu_info["total_memory"] * 0.95:
                            self.logger.warning("Using more than 95% of available memory. Stopping benchmark.")
                            break
                else:
                    self.logger.warning(f"Failed to run benchmark with {num_models} models. Stopping.")
                    break
            
            # Find sweet spot
            sweet_spot = self.find_sweet_spot()
            
            if sweet_spot:
                self.logger.info("\n" + "=" * 80)
                self.logger.info(f"BEST PERFORMING MODEL COUNT: {sweet_spot['sweet_spot']} models")
                self.logger.info(f"Processing time: {sweet_spot['min_processing_time']:.2f} seconds")
                if sweet_spot['min_frame_time'] is not None:
                    self.logger.info(f"Frame time: {sweet_spot['min_frame_time']:.2f} ms")
                self.logger.info("=" * 80)
            else:
                self.logger.warning("Could not determine sweet spot. No valid results.")
            
            # Generate plots
            plot_file = f"benchmark_plots_{timestamp}.png"
            self.plot_results(plot_file)
            
            # Prepare results
            benchmark_results = {
                "gpu_info": self.gpu_info,
                "results": self.results,
                "sweet_spot": sweet_spot,
                "plot_file": plot_file if self.results else None
            }
            
            # Save results to file
            results_file = f"benchmark_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(benchmark_results, f, indent=2)
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("Benchmark completed successfully!")
            self.logger.info(f"Results saved to: {results_file}")
            if self.results:
                self.logger.info(f"Plots saved to: {plot_file}")
            self.logger.info("=" * 80)
            
            return benchmark_results
            
        except Exception as e:
            self.logger.error(f"An error occurred during benchmarking: {str(e)}", exc_info=True)
            
            # Try to save partial results if we have any
            if self.results:
                try:
                    # Find sweet spot from partial results
                    sweet_spot = self.find_sweet_spot()
                    
                    # Generate plots from partial results
                    plot_file = f"benchmark_plots_{timestamp}_partial.png"
                    self.plot_results(plot_file)
                    
                    # Save partial results
                    partial_results = {
                        "gpu_info": self.gpu_info,
                        "results": self.results,
                        "sweet_spot": sweet_spot,
                        "plot_file": plot_file,
                        "error": str(e),
                        "status": "partial"
                    }
                    
                    results_file = f"benchmark_results_{timestamp}_partial.json"
                    with open(results_file, 'w') as f:
                        json.dump(partial_results, f, indent=2)
                    
                    self.logger.info(f"Partial results saved to: {results_file}")
                    self.logger.info(f"Partial plots saved to: {plot_file}")
                    
                    if sweet_spot:
                        self.logger.info("\n" + "=" * 80)
                        self.logger.info(f"BEST PERFORMING MODEL COUNT: {sweet_spot['sweet_spot']} models")
                        self.logger.info(f"Processing time: {sweet_spot['min_processing_time']:.2f} seconds")
                        if sweet_spot['min_frame_time'] is not None:
                            self.logger.info(f"Frame time: {sweet_spot['min_frame_time']:.2f} ms")
                        self.logger.info("=" * 80)
                    
                    return partial_results
                except Exception as save_error:
                    self.logger.error(f"Failed to save partial results: {str(save_error)}")
            
            raise 