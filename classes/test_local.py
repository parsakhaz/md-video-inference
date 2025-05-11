import argparse
import logging
import os
import time
from typing import Dict, Any, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from predictor import Predictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/test_local_{time.strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Default video URL (a short sample video)
DEFAULT_VIDEO_URL = "https://storage.googleapis.com/demo-videos/sample-5s.mp4"

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test the Moondream video analysis system locally")
    parser.add_argument("--video_url", type=str, default=DEFAULT_VIDEO_URL, help="URL of the video to analyze")
    parser.add_argument("--debug", action="store_true", help="Enable detailed timing logs")
    parser.add_argument("--fps", type=int, default=3, help="Number of frames to analyze per second")
    parser.add_argument("--output", type=str, default="results.txt", help="File to save results to")
    parser.add_argument("--num_models", type=int, default=3, help="Number of model instances to use")
    parser.add_argument("--use_queue", type=bool, default=True, help="Use queue-based processing")
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Create predictor
    predictor = Predictor()
    
    # Setup predictor with specified number of models
    predictor.setup(num_models=args.num_models)
    
    # Analyze video
    logging.info(f"Analyzing video: {args.video_url}")
    logging.info(f"Using {args.num_models} model instances")
    logging.info(f"Frames per second: {args.fps}")
    logging.info(f"Queue-based processing: {args.use_queue}")
    
    result = predictor.predict(
        video_url=args.video_url,
        debug_log=args.debug,
        use_queue=args.use_queue
    )
    
    # Extract results and timing metrics
    frame_descriptions = result["results"]
    timing_metrics = result["timing_metrics"]
    
    # Save results to file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"{os.path.splitext(args.output)[0]}_{timestamp}.txt"
    
    with open(output_file, "w") as f:
        f.write(f"Video Analysis Results for {args.video_url}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of models: {args.num_models}\n")
        f.write(f"Frames per second: {args.fps}\n")
        f.write(f"Queue-based processing: {args.use_queue}\n\n")
        
        f.write("Frame Descriptions:\n")
        f.write("==================\n")
        for timestamp_ms, description in frame_descriptions:
            f.write(f"[{timestamp_ms}ms] {description}\n")
        
        f.write("\nTiming Metrics:\n")
        f.write("==============\n")
        for name, value in timing_metrics.items():
            if isinstance(value, dict):
                f.write(f"{name}:\n")
                for k, v in value.items():
                    f.write(f"  {k}: {v:.2f}\n")
            else:
                f.write(f"{name}: {value:.2f}\n")
    
    logging.info(f"Results saved to {output_file}")
    
    # Print sample results
    logging.info("Sample Results:")
    logging.info("==============")
    for i, (timestamp_ms, description) in enumerate(frame_descriptions[:5]):
        logging.info(f"Frame {i+1} [{timestamp_ms}ms]: {description}")
    
    if len(frame_descriptions) > 5:
        logging.info(f"... and {len(frame_descriptions) - 5} more frames")

if __name__ == "__main__":
    main() 