#!/usr/bin/env python3
"""
Moondream Video Processing Client

This client allows for easy interaction with the Moondream Video Processing API.
It can submit one or multiple video processing jobs, poll for completion,
and save results in JSONL format.

Usage:
    python moondream_client.py --video "https://example.com/video.mp4" --fps 1
    python moondream_client.py --batch video_urls.txt --fps 1 --question "What is in this scene?"
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Union
import requests
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn


# Configuration
API_BASE_URL = "https://parsakhaz--moondream-video-processor-job-api.modal.run"
DEFAULT_OUTPUT_DIR = "results"
POLL_INTERVAL = 5  # seconds


console = Console()


def submit_job(video_url: str, fps: int = 1, question: str = "Describe this scene.") -> Dict:
    """Submit a video processing job to the API."""
    url = f"{API_BASE_URL}/api/submit"
    params = {
        "video_url": video_url,
        "target_fps": fps,
        "question": question
    }
    
    console.print(f"Submitting job for: [bold cyan]{video_url}[/bold cyan]")
    response = requests.post(url, params=params)
    
    if response.status_code != 200:
        console.print(f"[bold red]Error submitting job:[/bold red] {response.text}")
        return None
    
    result = response.json()
    console.print(f"Job submitted successfully. Job ID: [bold green]{result['job_id']}[/bold green]")
    return result


def get_job_status(job_id: str) -> Dict:
    """Get the status of a job."""
    url = f"{API_BASE_URL}/api/status/{job_id}"
    response = requests.get(url)
    
    if response.status_code != 200:
        console.print(f"[bold red]Error getting job status:[/bold red] {response.text}")
        return None
    
    return response.json()


def save_results_to_jsonl(job_data: Dict, output_file: str) -> None:
    """Save job results to a JSONL file, appending if the file exists."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    if job_data.get("status") != "completed":
        console.print(f"[bold yellow]Warning:[/bold yellow] Job not completed. Status: {job_data.get('status')}")
        return
    
    # Get frame results
    frame_results = job_data.get("results_payload", {}).get("frame_results", [])
    
    if not frame_results:
        console.print("[bold yellow]Warning:[/bold yellow] No frame results found")
        return
    
    # Add job_id to each frame result for traceability
    job_id = job_data.get("job_id")
    for result in frame_results:
        if "job_id" not in result:
            result["job_id"] = job_id
    
    # Write each frame result as a separate JSON line, appending to file if it exists
    with open(output_file, 'a') as f:
        for result in frame_results:
            f.write(json.dumps(result) + '\n')
    
    # Also save the full job data to a separate file with the job ID
    full_output_dir = os.path.join(os.path.dirname(output_file), "full_results")
    os.makedirs(full_output_dir, exist_ok=True)
    
    full_output_file = os.path.join(full_output_dir, f"{os.path.basename(output_file).split('.')[0]}_{job_id}.json")
    with open(full_output_file, 'w') as f:
        json.dump(job_data, f, indent=2)
    
    console.print(f"Appended [bold green]{len(frame_results)}[/bold green] frame results to [bold green]{output_file}[/bold green]")
    console.print(f"Saved full job data to [bold green]{full_output_file}[/bold green]")


def wait_for_job_completion(job_id: str, output_file: Optional[str] = None) -> Dict:
    """Poll until the job is completed and save results if output_file is provided."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(f"Waiting for job {job_id} to complete...", total=100)
        
        while True:
            job_data = get_job_status(job_id)
            
            if not job_data:
                progress.update(task, description=f"[bold red]Error getting job status for {job_id}[/bold red]")
                time.sleep(POLL_INTERVAL)
                continue
            
            status = job_data.get("status")
            progress_value = job_data.get("progress", 0)
            
            # Update progress description with status
            progress.update(
                task, 
                description=f"Job {job_id} status: {status.upper() if status else 'UNKNOWN'}", 
                completed=progress_value
            )
            
            # If job is complete or has failed, break the loop
            if status in ["completed", "failed"]:
                break
            
            # Sleep before polling again
            time.sleep(POLL_INTERVAL)
    
    # Save results if output file provided and job completed successfully
    if output_file and status == "completed":
        save_results_to_jsonl(job_data, output_file)
    
    return job_data


def process_video(video_url: str, fps: int, question: str, output_dir: str) -> None:
    """Process a single video: submit, wait for completion, and save results."""
    # Submit the job
    job_result = submit_job(video_url, fps, question)
    if not job_result:
        return
    
    job_id = job_result["job_id"]
    
    # Create an output filename based just on the video URL (not job-specific)
    video_name = os.path.basename(video_url.split("?")[0])  # Remove query params if any
    output_file = os.path.join(output_dir, f"{video_name}.jsonl")
    
    # Wait for completion and save results
    final_job_data = wait_for_job_completion(job_id, output_file)
    
    # Print summary
    if final_job_data:
        status = final_job_data.get("status")
        if status == "completed":
            frames_processed = len(final_job_data.get("results_payload", {}).get("frame_results", []))
            timings = final_job_data.get("results_payload", {}).get("timings_seconds", {})
            total_time = timings.get("overall_pipeline_internal", 0)
            
            console.print(f"\n[bold green]Job completed successfully![/bold green]")
            console.print(f"Processed [bold]{frames_processed}[/bold] frames in [bold]{total_time:.2f}[/bold] seconds")
            console.print(f"Results appended to [bold cyan]{output_file}[/bold cyan]")
        else:
            console.print(f"\n[bold red]Job failed![/bold red] Error: {final_job_data.get('error_message')}")


def process_batch(video_urls: List[str], fps: int, question: str, output_dir: str) -> None:
    """Process multiple videos in batch."""
    console.print(f"[bold]Processing {len(video_urls)} videos in batch:[/bold]")
    
    for i, video_url in enumerate(video_urls):
        console.print(f"\n[bold]Processing video {i+1}/{len(video_urls)}[/bold]")
        process_video(video_url, fps, question, output_dir)


def read_urls_from_file(file_path: str) -> List[str]:
    """Read URLs from a file, one per line."""
    with open(file_path, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    return urls


def main():
    parser = argparse.ArgumentParser(description="Moondream Video Processing Client")
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", type=str, help="URL of a single video to process")
    input_group.add_argument("--batch", type=str, help="File containing video URLs to process in batch (one per line)")
    
    # Processing parameters
    parser.add_argument("--fps", type=int, default=1, help="Frames per second to extract (default: 1)")
    parser.add_argument("--question", type=str, default="Describe this scene.", 
                       help="Question to ask about each frame (default: 'Describe this scene.')")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, 
                       help=f"Directory to save results (default: {DEFAULT_OUTPUT_DIR})")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process input based on command-line arguments
    if args.video:
        process_video(args.video, args.fps, args.question, args.output_dir)
    elif args.batch:
        video_urls = read_urls_from_file(args.batch)
        process_batch(video_urls, args.fps, args.question, args.output_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Process interrupted by user.[/bold yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1) 