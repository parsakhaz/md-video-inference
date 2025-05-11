#!/usr/bin/env python3
"""
Benchmark script to determine the optimal number of model instances for a given GPU.
This script tests different numbers of models and identifies the sweet spot based on performance metrics.
"""

import argparse
import sys
from classes.benchmark_utility import BenchmarkUtility

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Benchmark different numbers of model instances")
    parser.add_argument(
        "--video_url", 
        type=str, 
        default="https://videos.pexels.com/video-files/31554443/13448277_1080_1920_25fps.mp4",
        help="URL of the video to analyze"
    )
    parser.add_argument(
        "--max_models", 
        type=int, 
        default=10,
        help="Maximum number of models to test (default: 10)"
    )
    parser.add_argument(
        "--fps", 
        type=int, 
        default=3,
        help="Number of frames to analyze per second"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    return parser.parse_args()

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Create benchmark utility
        benchmark = BenchmarkUtility(debug_log=args.debug)
        
        # Run full benchmark suite
        benchmark.run_full_benchmark(
            video_url=args.video_url,
            max_models=args.max_models,
            fps=args.fps
        )
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 