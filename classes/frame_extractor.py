import cv2
import logging
import time
from typing import List, Dict, Any, Tuple

class FrameExtractor:
    """Handles extracting frames from videos at specified intervals"""
    
    def __init__(self):
        """Initialize the FrameExtractor"""
        self.frame_extraction_time = 0
        self.frames = []
        self.fps = 0
        
    def extract_frames(self, video_path: str, frames_per_second: int, debug_log: bool = False) -> Tuple[List[Dict[str, Any]], float, float]:
        """
        Extract frames from video at specified intervals
        
        Args:
            video_path: Path to the video file
            frames_per_second: Number of frames to extract per second
            debug_log: Whether to print detailed timing logs
            
        Returns:
            Tuple of (list of frames, video FPS, frame extraction time)
        """
        # Start timing the frame extraction
        frame_extraction_start = time.time()
        
        # Open video
        video_capture = cv2.VideoCapture(video_path)
        
        if not video_capture.isOpened():
            raise Exception("Error opening video file")
        
        # Get video properties
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_interval = int(self.fps) // frames_per_second
        if frame_interval < 1:
            frame_interval = 1
            
        logging.info(f"Extracting frames every {frame_interval} frames ({frames_per_second} FPS)")
        
        # Extract frames
        self.frames = []
        for frame_idx in range(0, total_frames, frame_interval):
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = video_capture.read()
            
            if not success:
                logging.warning(f"Failed to read frame at position {frame_idx}")
                continue
                
            self.frames.append({
                'frame_idx': frame_idx,
                'frame': frame.copy(),
                'fps': self.fps
            })
        
        # Release video capture
        video_capture.release()
        
        # Calculate frame extraction time
        self.frame_extraction_time = time.time() - frame_extraction_start
        if debug_log:
            logging.info(f"Frame extraction completed in {self.frame_extraction_time:.2f} seconds")
            logging.info(f"Extracted {len(self.frames)} frames")
            
        return self.frames, self.fps, self.frame_extraction_time
    
    def get_frames(self) -> List[Dict[str, Any]]:
        """Get the extracted frames"""
        return self.frames
    
    def get_fps(self) -> float:
        """Get the video FPS"""
        return self.fps
    
    def get_frame_extraction_time(self) -> float:
        """Get the time taken to extract frames"""
        return self.frame_extraction_time 