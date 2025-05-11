import torch
from PIL import Image
import cv2
import logging
import time
import math
from typing import Dict, Any, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

class FrameProcessor:
    """Handles processing frames with the Moondream model"""
    
    def __init__(self, models: List[Any], tokenizer: Any, debug_log: bool = False):
        """
        Initialize the FrameProcessor
        
        Args:
            models: List of Moondream model instances
            tokenizer: Moondream tokenizer
            debug_log: Whether to print detailed timing logs
        """
        self.models = models
        self.tokenizer = tokenizer
        self.debug_log = debug_log
        self.num_models = len(models)
        self.processing_time = 0
        self.encoding_times = []
        self.inference_times = []
        self.results = {}
        
        # Verify that models are loaded on GPU
        if debug_log:
            for i, model in enumerate(models):
                logging.info(f"FrameProcessor: Moondream model {i+1} is on device: {next(model.parameters()).device}")
    
    def process_frame(self, frame_data: Dict[str, Any]) -> Tuple[int, str, float, float]:
        """
        Process a single frame with the specified model
        
        Args:
            frame_data: Dictionary containing frame data and model information
            
        Returns:
            Tuple of (timestamp, description, encoding time, inference time)
        """
        frame_idx = frame_data['frame_idx']
        frame = frame_data['frame']
        model = frame_data['model']
        tokenizer = frame_data['tokenizer']
        fps = frame_data['fps']
        model_name = frame_data['model_name']
        debug_log = frame_data.get('debug_log', False)
        
        # Convert frame to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Encode image
        start_encode_time = time.time()
        enc_image = model.encode_image(image)
        encode_time_ms = (time.time() - start_encode_time) * 1000
        if debug_log:
            logging.info(f"Frame {frame_idx} image encoding time ({model_name}): {encode_time_ms:.2f} ms")
        
        # Inference
        start_inference_time = time.time()
        moondream_description = model.answer_question(enc_image, "Describe as if captioning", tokenizer)
        inference_time_ms = (time.time() - start_inference_time) * 1000
        if debug_log:
            logging.info(f"Frame {frame_idx} model inference time ({model_name}): {inference_time_ms:.2f} ms")
        
        # Calculate timestamp
        timestamp = frame_idx / fps * 1000
        rounded_timestamp = math.floor(timestamp)
        
        return (rounded_timestamp, moondream_description, encode_time_ms, inference_time_ms)
    
    def process_frames(self, frames: List[Dict[str, Any]]) -> List[Tuple[int, str]]:
        """
        Process a list of frames in parallel
        
        Args:
            frames: List of frame dictionaries
            
        Returns:
            List of tuples (timestamp, description)
        """
        # Start timing the processing
        processing_start = time.time()
        
        # Prepare frames with model information
        frame_tasks = []
        model_idx = 0
        
        for frame in frames:
            current_model = self.models[model_idx % self.num_models]
            model_name = f"Model {model_idx % self.num_models + 1}"
            
            frame_data = {
                'frame_idx': frame['frame_idx'],
                'frame': frame['frame'],
                'model': current_model,
                'tokenizer': self.tokenizer,
                'fps': frame['fps'],
                'model_name': model_name,
                'debug_log': self.debug_log
            }
            
            frame_tasks.append(frame_data)
            model_idx += 1
        
        # Process frames in parallel
        self.results = {}
        
        with ThreadPoolExecutor(max_workers=self.num_models) as executor:
            future_to_idx = {executor.submit(self.process_frame, task): task['frame_idx'] for task in frame_tasks}
            
            for future in as_completed(future_to_idx):
                frame_idx = future_to_idx[future]
                try:
                    timestamp, description, encode_time, inference_time = future.result()
                    self.results[frame_idx] = (timestamp, description)
                    self.encoding_times.append(encode_time)
                    self.inference_times.append(inference_time)
                except Exception as e:
                    logging.error(f"Error processing frame {frame_idx}: {str(e)}")
                    raise
        
        # Calculate processing time
        self.processing_time = time.time() - processing_start
        
        # Sort results by frame index
        sorted_results = [self.results[idx] for idx in sorted(self.results.keys())]
        
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