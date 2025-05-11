import queue
import threading
from typing import List, Dict, Any, Optional
import logging

class FrameQueue:
    """A thread-safe queue for managing video frame processing tasks"""
    
    def __init__(self):
        self.queue = queue.Queue()
        self.results = {}
        self._lock = threading.Lock()
        
    def add_frames(self, frames: List[Dict[str, Any]]) -> None:
        """Add frames to the queue"""
        for frame in frames:
            self.queue.put(frame)
            
    def get_next_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Get next batch of frames from queue"""
        batch = []
        try:
            for _ in range(batch_size):
                frame = self.queue.get_nowait()
                batch.append(frame)
        except queue.Empty:
            pass
        return batch
    
    def update_frame_result(self, frame_idx: int, result: tuple) -> None:
        """Update the result for a processed frame"""
        with self._lock:
            self.results[frame_idx] = result
            
    def get_results(self) -> List[tuple]:
        """Get all results sorted by frame index"""
        with self._lock:
            return [self.results[idx] for idx in sorted(self.results.keys())]
            
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return self.queue.empty()
        
    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self.queue.qsize() 