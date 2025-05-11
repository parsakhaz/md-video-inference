import os
import requests
import uuid
import tempfile
import logging
import time
from typing import Tuple, Optional

class VideoDownloader:
    """Handles downloading videos from URLs and storing them temporarily"""
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize the VideoDownloader
        
        Args:
            temp_dir: Optional temporary directory to use for downloads
        """
        self.temp_dir = temp_dir
        self.download_time = 0
        self.video_path = None
        
    def download(self, video_url: str, debug_log: bool = False) -> Tuple[str, float]:
        """
        Download a video from a URL
        
        Args:
            video_url: URL of the video to download
            debug_log: Whether to print detailed timing logs
            
        Returns:
            Tuple of (path to downloaded video, download time in seconds)
        """
        # Start timing the download
        download_start_time = time.time()
        
        # Create a temporary directory if not provided
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp()
            
        # Download video
        logging.info("Downloading video")
        response = requests.get(video_url)
        if response.status_code == 200:
            filename = os.path.join(self.temp_dir, f"{str(uuid.uuid4())}.mp4")
            with open(filename, "wb") as f:
                f.write(response.content)
            self.download_time = time.time() - download_start_time
            self.video_path = filename
            
            if debug_log:
                logging.info(f"Video download completed in {self.download_time:.2f} seconds")
                
            return filename, self.download_time
        else:
            raise Exception(f"Error downloading video: {response.status_code}")
            
    def cleanup(self) -> None:
        """Clean up downloaded video file"""
        if self.video_path and os.path.exists(self.video_path):
            try:
                os.remove(self.video_path)
            except Exception as e:
                logging.warning(f"Error cleaning up video file: {str(e)}")
                
    def get_download_time(self) -> float:
        """Get the time taken to download the video"""
        return self.download_time 