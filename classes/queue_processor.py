import logging
import time
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from .frame_queue import FrameQueue
from .frame_processor import FrameProcessor

class QueueProcessor:
    """Handles queue-based processing of frames"""
    
    def __init__(self, frame_processor: FrameProcessor, debug_log: bool = False):
        """
        Initialize the QueueProcessor
        
        Args:
            frame_processor: FrameProcessor instance
            debug_log: Whether to print detailed timing logs
        """
        self.frame_processor = frame_processor
        self.debug_log = debug_log
        self.processing_time = 0
        self.encoding_times = []
        self.inference_times = []
        
    def process_frames_with_queue(self, frames: List[Dict[str, Any]]) -> List[Tuple[int, str]]:
        """
        Process frames using a queue
        
        Args:
            frames: List of frame dictionaries
            
        Returns:
            List of tuples (timestamp, description)
        """
        # Start timing the processing
        processing_start = time.time()
        
        # Create frame queue
        frame_queue = FrameQueue()
        
        # Prepare frames with model information
        for i, frame in enumerate(frames):
            model_idx = i % self.frame_processor.num_models
            frame.update({
                'model': self.frame_processor.models[model_idx],
                'tokenizer': self.frame_processor.tokenizer,
                'model_name': f"Model {model_idx + 1}",
                'debug_log': self.debug_log
            })
        
        # Add frames to queue
        frame_queue.add_frames(frames)
        
        # Process frames using queue
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.frame_processor.num_models) as executor:
            while not frame_queue.is_empty():
                batch = frame_queue.get_next_batch(self.frame_processor.num_models)
                futures = []
                
                for frame_data in batch:
                    future = executor.submit(self.frame_processor.process_frame, frame_data)
                    futures.append((future, frame_data['frame_idx']))
                
                for future, frame_idx in futures:
                    try:
                        timestamp, description, encode_time, inference_time = future.result()
                        frame_queue.update_frame_result(frame_idx, (timestamp, description))
                        self.encoding_times.append(encode_time)
                        self.inference_times.append(inference_time)
                    except Exception as e:
                        logging.error(f"Error processing frame {frame_idx}: {str(e)}")
                        raise
        
        # Calculate processing time
        self.processing_time = time.time() - processing_start
        
        # Get sorted results
        sorted_results = frame_queue.get_results()
        
        return sorted_results
    
    def get_processing_time(self) -> float:
        """Get the time taken to process frames"""
        return self.processing_time
    
    def get_encoding_times(self) -> List[float]:
        """Get the encoding times for all frames"""
        return self.encoding_times
    
    def get_inference_times(self) -> List[float]:
        """Get the inference times for all frames"""
        return self.inference_times 