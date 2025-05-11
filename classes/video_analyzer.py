import logging
import time
from typing import List, Dict, Any, Tuple
from .video_downloader import VideoDownloader
from .frame_extractor import FrameExtractor
from .frame_processor import FrameProcessor
from .queue_processor import QueueProcessor
from .timing_metrics import TimingMetrics

class VideoAnalyzer:
    """Orchestrates the entire video analysis process"""
    
    def __init__(self, models: List[Any], tokenizer: Any, debug_log: bool = False):
        """
        Initialize the VideoAnalyzer
        
        Args:
            models: List of Moondream model instances
            tokenizer: Moondream tokenizer
            debug_log: Whether to print detailed timing logs
        """
        self.models = models
        self.tokenizer = tokenizer
        self.debug_log = debug_log
        self.num_models = len(models)
        
        # Initialize components
        self.video_downloader = VideoDownloader()
        self.frame_extractor = FrameExtractor()
        self.frame_processor = FrameProcessor(models, tokenizer, debug_log)
        self.queue_processor = QueueProcessor(self.frame_processor, debug_log)
        self.timing_metrics = TimingMetrics()
        
    def analyze_video(
        self, 
        video_url: str, 
        frames_per_second: int = 3, 
        use_queue: bool = True
    ) -> Tuple[List[Tuple[int, str]], Dict[str, Any]]:
        """
        Analyze a video and return frame descriptions
        
        Args:
            video_url: URL of the video to analyze
            frames_per_second: Number of frames to analyze per second
            use_queue: Whether to use queue-based processing
            
        Returns:
            Tuple of (list of tuples (timestamp, description), timing metrics)
        """
        # Start timing the entire process
        process_start_time = time.time()
        
        try:
            # Download video
            video_path, download_time = self.video_downloader.download(video_url, self.debug_log)
            self.timing_metrics.add_metric("Download time", download_time)
            
            # Extract frames
            frames, fps, frame_extraction_time = self.frame_extractor.extract_frames(
                video_path, frames_per_second, self.debug_log
            )
            self.timing_metrics.add_metric("Frame extraction time", frame_extraction_time)
            
            # Process frames
            if use_queue:
                # Use queue-based processing
                sorted_results = self.queue_processor.process_frames_with_queue(frames)
                processing_time = self.queue_processor.get_processing_time()
                self.timing_metrics.add_encoding_times(self.queue_processor.get_encoding_times())
                self.timing_metrics.add_inference_times(self.queue_processor.get_inference_times())
            else:
                # Use direct processing
                sorted_results = self.frame_processor.process_frames(frames)
                processing_time = self.frame_processor.get_processing_time()
                self.timing_metrics.add_encoding_times(self.frame_processor.get_encoding_times())
                self.timing_metrics.add_inference_times(self.frame_processor.get_inference_times())
                
            self.timing_metrics.add_metric("Processing time", processing_time)
            
            # Print timing summary
            if self.debug_log:
                self.timing_metrics.print_summary(len(sorted_results))
                
            # Return results and timing metrics
            return sorted_results, self.timing_metrics.get_summary_dict(len(sorted_results))
            
        except Exception as e:
            logging.error(f"An error occurred during video analysis: {str(e)}")
            raise
            
        finally:
            # Clean up downloaded video
            self.video_downloader.cleanup() 