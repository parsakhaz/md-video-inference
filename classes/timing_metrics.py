import logging
import time
from typing import Dict, Any, List

class TimingMetrics:
    """Handles collecting and reporting timing metrics"""
    
    def __init__(self):
        """Initialize the TimingMetrics"""
        self.start_time = time.time()
        self.metrics = {}
        self.encoding_times = []
        self.inference_times = []
        
    def add_metric(self, name: str, value: float) -> None:
        """
        Add a timing metric
        
        Args:
            name: Name of the metric
            value: Value of the metric in seconds
        """
        self.metrics[name] = value
        
    def add_encoding_times(self, times: List[float]) -> None:
        """
        Add encoding times
        
        Args:
            times: List of encoding times in milliseconds
        """
        self.encoding_times.extend(times)
        
    def add_inference_times(self, times: List[float]) -> None:
        """
        Add inference times
        
        Args:
            times: List of inference times in milliseconds
        """
        self.inference_times.extend(times)
        
    def get_total_time(self) -> float:
        """Get the total time since initialization"""
        return time.time() - self.start_time
        
    def get_metrics(self) -> Dict[str, float]:
        """Get all timing metrics"""
        return self.metrics
        
    def get_encoding_times(self) -> List[float]:
        """Get all encoding times"""
        return self.encoding_times
        
    def get_inference_times(self) -> List[float]:
        """Get all inference times"""
        return self.inference_times
        
    def get_encoding_stats(self) -> Dict[str, float]:
        """Get encoding time statistics"""
        if not self.encoding_times:
            return {"avg": 0, "min": 0, "max": 0}
            
        return {
            "avg": sum(self.encoding_times) / len(self.encoding_times),
            "min": min(self.encoding_times),
            "max": max(self.encoding_times)
        }
        
    def get_inference_stats(self) -> Dict[str, float]:
        """Get inference time statistics"""
        if not self.inference_times:
            return {"avg": 0, "min": 0, "max": 0}
            
        return {
            "avg": sum(self.inference_times) / len(self.inference_times),
            "min": min(self.inference_times),
            "max": max(self.inference_times)
        }
        
    def print_summary(self, num_frames: int) -> None:
        """
        Print a summary of all timing metrics
        
        Args:
            num_frames: Number of frames processed
        """
        total_time = self.get_total_time()
        encoding_stats = self.get_encoding_stats()
        inference_stats = self.get_inference_stats()
        
        logging.info(f"======= Video Processing Timing Summary =======")
        logging.info(f"Total frames analyzed: {num_frames}")
        
        for name, value in self.metrics.items():
            logging.info(f"{name}: {value:.2f} seconds")
            
        logging.info(f"Encoding time (avg/min/max): {encoding_stats['avg']:.2f}/{encoding_stats['min']:.2f}/{encoding_stats['max']:.2f} ms")
        logging.info(f"Inference time (avg/min/max): {inference_stats['avg']:.2f}/{inference_stats['min']:.2f}/{inference_stats['max']:.2f} ms")
        logging.info(f"Total end-to-end time: {total_time:.2f} seconds")
        logging.info(f"Average time per frame: {(total_time / num_frames * 1000):.2f} ms")
        logging.info(f"===========================================")
        
    def get_summary_dict(self, num_frames: int) -> Dict[str, Any]:
        """
        Get a dictionary summary of all timing metrics
        
        Args:
            num_frames: Number of frames processed
            
        Returns:
            Dictionary containing all timing metrics
        """
        total_time = self.get_total_time()
        encoding_stats = self.get_encoding_stats()
        inference_stats = self.get_inference_stats()
        
        return {
            "total_frames": num_frames,
            "metrics": self.metrics,
            "encoding_stats": encoding_stats,
            "inference_stats": inference_stats,
            "total_time": total_time,
            "avg_time_per_frame_ms": (total_time / num_frames * 1000) if num_frames > 0 else 0
        } 