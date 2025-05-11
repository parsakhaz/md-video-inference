# Moondream Video Processing API

A Modal-based API for asynchronous video processing and analysis using the Moondream2 vision-language model. This system processes videos by extracting frames at a specified rate and generating descriptive text for each frame.

## Features

- Asynchronous job processing system for video analysis
- RESTful API endpoints for easy integration
- Persisted job history with detailed status tracking
- Configurable frame extraction rate
- Customizable questions for the vision-language model
- GPU-accelerated inference using Modal cloud infrastructure

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
   - Distributes frame analysis across GPU workers
   - Collects and aggregates results
4. **Completion**: Job is marked as "completed" with full results or "failed" with error details
5. **Results Retrieval**: Client fetches job status and results using the job ID

## Development and Customization

### Modify the Model

To use a different model, update the `MODEL_ID` and `REVISION` constants at the top of the file. Ensure the model has compatible methods for vision-language tasks.

### Adjust GPU Resources

The application uses A10G GPUs by default. You can modify the GPU type and concurrency limits in the `@app.cls` decorator of the `MoondreamWorker` class.

### Determining Optimal Concurrency Limit

To maximize the efficiency of your GPU resources, you should determine the optimal concurrency limit for your specific GPU model. You can use the following benchmark script to find this sweet spot.

Run: benchmark_gpu_concurrency.py to benchmark. The classes are to allow the benchmark script to run. They are not used for modal.

#### Running the Benchmark

1. SSH into a GPU instance (e.g., using services like RunPod or AWS EC2 with GPU)
2. Install the necessary dependencies:
   ```bash
   pip install torch transformers pillow opencv-python numpy
   ```
3. Run the benchmark script:
   ```bash
   python benchmark_gpu_concurrency.py --max_models 20 --fps 2
   ```

The script will test different concurrency levels and output the optimal number based on total throughput and latency. Once you have determined the best value, update the `concurrency_limit` parameter in the `MoondreamWorker` class:

```python
@app.cls(gpu="a10g", timeout=180, image=moondream_image, concurrency_limit=YOUR_OPTIMAL_VALUE)
class MoondreamWorker:
    # ... rest of the class
```

#### Expected Results

Different GPU models will have different optimal concurrency values:
- NVIDIA A10G: Typically 10-15 concurrent models
- NVIDIA A100: Can handle 20-30 concurrent models
- NVIDIA T4: Usually 5-10 concurrent models

The optimal value balances memory usage and computational efficiency. Setting too high a value may cause out-of-memory errors, while too low a value underutilizes the GPU.

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