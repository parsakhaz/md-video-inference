# Moondream Video Processing API

A Modal-based API for asynchronous video processing and analysis using the Moondream2 vision-language model. This system processes videos by extracting frames at a specified rate and generating descriptive text for each frame.

## Features

- Asynchronous job processing system for video analysis
- RESTful API endpoints for easy integration
- Persisted job history with detailed status tracking
- Configurable frame extraction rate
- Customizable questions for the vision-language model
- GPU-accelerated inference using Modal cloud infrastructure
- High parallelism with up to 30 concurrent GPU workers
- Real-time job progress tracking
- Python client for easy batch processing (see `/client` directory)

## Prerequisites

- Python 3.10+
- [Modal](https://modal.com/) account
- Internet connection to access video URLs

## Quick Start

### 1. Install Required Dependencies

```bash
pip install modal
```

### 2. Configure Modal

If you haven't already, set up Modal:

```bash
modal token new
```

### 3. Deploy the API

```bash
# Deploy to Modal's infrastructure
modal deploy moondream_video_processor.py
```

### 4. Use the API

After deployment, your API will be available at the URL provided by Modal.

## API Endpoints

### Submit a Video Processing Job

```
POST /api/submit
```

Parameters:
- `video_url` (required): URL of the video to process
- `target_fps` (optional, default=1): Number of frames per second to extract
- `question` (optional, default="Describe this scene."): Question to ask about each frame

Example Request:
```bash
curl -X POST "https://your-modal-app.modal.run/api/submit" \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://example.com/video.mp4", "target_fps": 1, "question": "What objects are visible?"}'
```

Example Response:
```json
{
  "job_id": "job_123e4567-e89b-12d3-a456-426614174000",
  "status": "submitted",
  "message": "Job submitted for processing."
}
```

### Check Job Status

```
GET /api/status/{job_id}
```

Example Request:
```bash
curl "https://your-modal-app.modal.run/api/status/job_123e4567-e89b-12d3-a456-426614174000"
```

Example Response:
```json
{
  "job_id": "job_123e4567-e89b-12d3-a456-426614174000",
  "status": "completed",
  "video_url": "https://example.com/video.mp4",
  "input_target_fps": 1,
  "input_question": "What objects are visible?",
  "submitted_at": 1684123456.789,
  "completed_at": 1684123556.789,
  "results_payload": {
    "video_url": "https://example.com/video.mp4",
    "target_fps_setting": 1,
    "actual_original_fps": 30,
    "total_frames_submitted_to_workers": 10,
    "total_frames_results_received": 10,
    "timings_seconds": {
      "video_download": 2.45,
      "frame_extraction": 1.23,
      "distributed_frame_processing_map": 15.67,
      "overall_pipeline_internal": 19.35
    },
    "frame_results": [
      {
        "job_id_for_frame": "job_123e4567-e89b-12d3-a456-426614174000",
        "frame_idx": 0,
        "timestamp_ms": 0,
        "description": "A red car parked on a street next to a building.",
        "processing_times_ms": {
          "image_encode": 245.32,
          "llm_inference": 1223.45,
          "total_worker_frame_time": 1468.77
        },
        "status": "success"
      },
      // ... more frame results
    ]
  }
}
```

### List Recent Jobs

```
GET /api/jobs/list
```

Parameters:
- `limit` (optional, default=10): Maximum number of jobs to return
- `status_filter` (optional): Filter by job status (e.g., "completed", "failed")

Example Request:
```bash
curl "https://your-modal-app.modal.run/api/jobs/list?limit=5&status_filter=completed"
```

Example Response:
```json
{
  "jobs": [
    {
      "job_id": "job_123e4567-e89b-12d3-a456-426614174000",
      "video_url": "https://example.com/video1.mp4",
      "status": "completed",
      "submitted_at": 1684123456.789,
      "completed_at": 1684123556.789
    },
    // ... more jobs
  ],
  "total_count": 5
}
```

## Results Storage and Analysis

### Storing Results in JSONL Format

The API provides detailed frame-by-frame results that can be stored for further analysis. To save the results in a JSONL (JSON Lines) format, which is convenient for processing large datasets, you can use the following script:

```bash
#!/bin/bash
# Usage: ./save_results.sh JOB_ID output_file.jsonl

JOB_ID=$1
OUTPUT_FILE=$2

# Check if output file exists and create it if not
touch "$OUTPUT_FILE"

# Get the full job status response
RESPONSE=$(curl -s "https://YOUR_WORKSPACE--moondream-video-processor-job-api.modal.run/api/status/$JOB_ID")

# Extract frame results from the response and save each as a separate line in JSONL
echo "$RESPONSE" | jq -c '.results_payload.frame_results[]' >> "$OUTPUT_FILE"

echo "Saved $(jq -s 'length' "$OUTPUT_FILE") frame results to $OUTPUT_FILE"
```

Replace `YOUR_WORKSPACE` with your Modal workspace name.

You can also use Python to retrieve and analyze the results:

```python
import requests
import json
import pandas as pd

def save_job_results_to_jsonl(job_id, output_file):
    """Save job results to JSONL file for easy analysis."""
    api_url = f"https://YOUR_WORKSPACE--moondream-video-processor-job-api.modal.run/api/status/{job_id}"
    
    response = requests.get(api_url)
    if response.status_code != 200:
        raise Exception(f"Failed to get job status: {response.text}")
    
    job_data = response.json()
    
    if job_data["status"] != "completed":
        raise Exception(f"Job not completed yet. Current status: {job_data['status']}")
    
    # Extract frame results
    frame_results = job_data["results_payload"]["frame_results"]
    
    # Write each frame result as a separate JSON line
    with open(output_file, 'w') as f:
        for result in frame_results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Saved {len(frame_results)} frame results to {output_file}")
    return frame_results

# Example usage
job_id = "job_e8cfe37b-ceb5-40e1-9338-1b0426d262e8"
save_job_results_to_jsonl(job_id, "video_analysis_results.jsonl")

# To analyze the results with pandas
def analyze_results(jsonl_file):
    """Load JSONL results into pandas for analysis."""
    df = pd.read_json(jsonl_file, lines=True)
    
    # Example analysis
    print(f"Total frames: {len(df)}")
    print(f"Average processing time: {df['processing_times_ms'].apply(lambda x: x['total_worker_frame_time']).mean():.2f} ms")
    
    # Extract just the descriptions
    descriptions = df[['timestamp_ms', 'description']]
    return df, descriptions

df, descriptions = analyze_results("video_analysis_results.jsonl")
```

This approach makes it easy to store and process the frame-by-frame analysis results, enabling further analysis with tools like pandas, visualization libraries, or text analysis frameworks.

## Running Locally

To run the system locally for testing:

```bash
# Run with default test video
modal run moondream_video_processor.py

# Or with custom parameters
modal run moondream_video_processor.py \
  --video-url "https://example.com/video.mp4" \
  --target-fps 2 \
  --question "What objects are visible?" \
  --output-json-file "results.json"
```

## Job Processing Flow

1. **Job Submission**: Client submits a video URL for processing
2. **Initial Status**: Job is marked as "submitted" and assigned a unique ID
3. **Processing**: System asynchronously:
   - Downloads the video
   - Extracts frames at the specified FPS
   - Distributes frame analysis across GPU workers in batches
   - Tracks and reports job progress
   - Collects and aggregates results
4. **Completion**: Job is marked as "completed" with full results or "failed" with error details
5. **Results Retrieval**: Client fetches job status and results using the job ID

## Development and Customization

### Modify the Model

To use a different model, update the `MODEL_ID` and `REVISION` constants at the top of the file. Ensure the model has compatible methods for vision-language tasks.

### Adjust GPU Resources

The application uses A10G GPUs by default. You can modify the GPU type and parallelism settings in the `@app.cls` decorator of the `MoondreamWorker` class.

```python
@app.cls(gpu="a10g", timeout=180, image=moondream_image, max_containers=30, min_containers=0)
class MoondreamWorker:
    # ... rest of the class
```

The `max_containers` parameter controls how many parallel GPU workers can be used, while `min_containers` keeps a minimum number of workers warm to reduce cold start times.

### Determining Optimal Concurrency Limit

The system has two levels of concurrency to understand:

1. **Per-GPU Model Concurrency**: How many inference requests a single GPU worker can handle simultaneously. This is set by the `concurrency_limit` parameter in the `@modal.concurrent` decorator (not currently used in our implementation).

2. **Total Worker Parallelism**: How many separate GPU workers can be created to process frames in parallel. This is controlled by the `max_containers` parameter.

The benchmark script is designed to determine the optimal number of concurrent model instances per GPU worker (the first type), not the total number of workers. Our implementation currently uses a simpler approach where each worker processes one request at a time, but we scale out to many workers.

```python
# Current implementation scales horizontally with many workers
@app.cls(gpu="a10g", timeout=180, image=moondream_image, max_containers=30, min_containers=0)
class MoondreamWorker:
    # ... rest of the class
```

If you want to optimize further, you could implement model batching within each GPU worker using the benchmark script to determine the optimal batch size per GPU.

#### Expected Results

Different GPU models will have different optimal values:
- NVIDIA A10G: Typically can handle 3-5 concurrent model instances per GPU
- NVIDIA A100: Can handle 5-10 concurrent model instances per GPU
- NVIDIA T4: Usually 2-3 concurrent model instances per GPU

The current implementation foregoes batching within workers and instead focuses on horizontal scaling with many workers processing frames in parallel.

### Extend the API

To add new endpoints, follow the pattern used for existing web endpoints, adding validation and error handling as needed.

## Troubleshooting

### Common Error Messages

- **"Invalid video URL"**: The provided URL does not use the http/https protocol
- **"No frames extracted from video"**: The video could not be processed or is empty
- **"Job ID not found in store"**: The specified job ID does not exist or has been purged

### Checking Logs

To view logs for your deployed application:

```bash
modal logs moondream-video-processor-job
```

## License

This project uses the Moondream2 model which has its own license terms. Please check the license of the Moondream2 model at [vikhyatk/moondream2](https://huggingface.co/vikhyatk/moondream2) before using this application.

## Acknowledgments

- [Moondream2](https://huggingface.co/vikhyatk/moondream2) for the vision-language model
- [Modal](https://modal.com/) for the serverless compute platform

## Client

A Python client is included in the `/client` directory to make it easy to interact with the API. The client supports:

- Processing single videos or batches of videos
- Automatically polling for job completion
- Saving results in JSONL format for further analysis
- Real-time progress reporting

See the `/client/README.md` for detailed usage instructions and examples. 