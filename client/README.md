# Moondream Video Processing Client

A command-line client for interacting with the Moondream Video Processing API.

## Features

- Submit single video processing jobs
- Process multiple videos in batch mode
- Poll for job completion with nice progress display
- Automatically save results in JSONL format
- Save complete job data separately

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Process a Single Video

```bash
python moondream_client.py --video "https://example.com/video.mp4" --fps 1
```

### Process Multiple Videos in Batch

Create a text file with video URLs (one per line):

```
# video_urls.txt
https://example.com/video1.mp4
https://example.com/video2.mp4
https://example.com/video3.mp4
```

Then run:

```bash
python moondream_client.py --batch video_urls.txt --fps 1 --question "What is happening in this scene?"
```

### Command-Line Options

- `--video URL`: Process a single video at the given URL
- `--batch FILE`: Process multiple videos from URLs in the given file (one per line)
- `--fps N`: Extract N frames per second (default: 1)
- `--question TEXT`: Question to ask about each frame (default: "Describe this scene.")
- `--output-dir DIR`: Directory to save results (default: "results")
- `--output-file FILE`: Filename for results (default: "results.jsonl")

## API Specification

### API Endpoints

The client interacts with the following API endpoints:

#### Submit a Job

```
POST /api/submit
```

Parameters:
- `video_url` (string): URL of the video to process
- `target_fps` (number): Frames per second to extract
- `question` (string): Question to ask about each frame

Example Request:
```python
response = requests.post(
    "https://parsakhaz--moondream-video-processor-job-api.modal.run/api/submit",
    params={
        "video_url": "https://example.com/video.mp4",
        "target_fps": 1,
        "question": "Describe this scene."
    }
)
```

Example Response:
```json
{
    "job_id": "job_12345678-abcd-efgh-ijkl-9876543210ab",
    "status": "submitted",
    "created_at": "2023-05-10T21:35:47.123456",
    "video_url": "https://example.com/video.mp4",
    "target_fps": 1,
    "question": "Describe this scene."
}
```

#### Get Job Status

```
GET /api/status/{job_id}
```

Example Request:
```python
response = requests.get(
    "https://parsakhaz--moondream-video-processor-job-api.modal.run/api/status/job_12345678-abcd-efgh-ijkl-9876543210ab"
)
```

Example Response (In Progress):
```json
{
    "job_id": "job_12345678-abcd-efgh-ijkl-9876543210ab",
    "status": "processing",
    "progress": 45,
    "created_at": "2023-05-10T21:35:47.123456",
    "video_url": "https://example.com/video.mp4"
}
```

Example Response (Completed):
```json
{
    "job_id": "job_12345678-abcd-efgh-ijkl-9876543210ab",
    "status": "completed",
    "progress": 100,
    "created_at": "2023-05-10T21:35:47.123456",
    "completed_at": "2023-05-10T21:36:12.987654",
    "video_url": "https://example.com/video.mp4",
    "results_payload": {
        "frame_results": [
            {
                "job_id_for_frame": "job_12345678-abcd-efgh-ijkl-9876543210ab",
                "frame_idx": 0,
                "timestamp_ms": 0,
                "description": "A person walking on a street with buildings in the background.",
                "processing_times_ms": {
                    "image_encode": 25.32,
                    "llm_inference": 342.18,
                    "total_worker_frame_time": 372.64
                },
                "status": "success"
            },
            // Additional frames...
        ],
        "timings_seconds": {
            "overall_pipeline_internal": 12.34,
            "video_download": 0.89,
            "frame_extraction": 1.23,
            "inference": 10.22
        },
        "metadata": {
            "video_duration_seconds": 60.0,
            "frame_count": 60,
            "target_fps": 1,
            "question": "Describe this scene."
        }
    }
}
```

## Output Format

### JSONL Output Structure

The client saves all frame results to a single JSONL file (default: `results/results.jsonl`). Each line in the file is a JSON object representing one frame with the following structure:

```json
{
    "job_id_for_frame": "job_12345678-abcd-efgh-ijkl-9876543210ab",
    "frame_idx": 0,
    "timestamp_ms": 0,
    "description": "A person walking on a street with buildings in the background.",
    "processing_times_ms": {
        "image_encode": 25.32,
        "llm_inference": 342.18,
        "total_worker_frame_time": 372.64
    },
    "status": "success",
    "job_id": "job_12345678-abcd-efgh-ijkl-9876543210ab",
    "video_url": "https://example.com/video.mp4"
}
```

Fields:
- `job_id_for_frame`: The ID of the job that processed this frame (matches the API response job_id)
- `job_id`: The same as job_id_for_frame, duplicated for traceability
- `video_url`: The URL of the video this frame was extracted from
- `frame_idx`: The frame's index (0-based) in the extraction sequence
- `timestamp_ms`: The frame's timestamp in milliseconds
- `description`: The AI-generated description based on the provided question
- `processing_times_ms`: Detailed timing information for this frame
- `status`: Processing status for this frame ("success" or error message)

### Full JSON Output Structure

For each job, the client also saves a complete record of the job data in the `results/full_results/` directory, with a filename pattern of `job_{job_id}.json`. This file includes all the information returned by the API, including frame results and overall job metadata.

## Performance

Based on our tests, here are the actual processing times with warm containers (no cold start):

| Video                                  | Duration | Frames Processed | Processing Time | FPS Setting |
|----------------------------------------|----------|------------------|----------------|------------|
| 214409_tiny.mp4 (cloud timelapse)      | 50 sec   | 251 frames       | 47.62 seconds  | 5 fps      |
| Big_Buck_Bunny_360_10s_1MB.mp4         | 10 sec   | 50 frames        | 4.68 seconds   | 5 fps      |
| ForBiggerBlazes.mp4                    | 15 sec   | 72 frames        | 6.15 seconds   | 5 fps      |

Processing time depends on:
- Cold start time (first request after deployment can take 20-30s)
- Number of frames extracted (based on FPS setting)
- Model inference time per frame (~1.5s per frame with Moondream2)
- Current load on the Modal infrastructure
- Video complexity
- Parallel processing (with GPU workers, processing scales well with more frames)

## Examples

### Single Video Processing

```bash
# Process a single video with default parameters
python moondream_client.py --video "https://cdn.pixabay.com/video/2024/05/29/214409_tiny.mp4"

# Process with custom FPS and question
python moondream_client.py --video "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4" --fps 2 --question "What animals are visible in this scene?"

# Save results to a custom directory and file
python moondream_client.py --video "https://example.com/my_video.mp4" --output-dir "my_results" --output-file "custom_results.jsonl"
```

### Batch Processing

```bash
# Create a file with video URLs
echo "https://cdn.pixabay.com/video/2024/05/29/214409_tiny.mp4
https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4
https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4" > videos_to_process.txt

# Process all videos in batch
python moondream_client.py --batch videos_to_process.txt --fps 5 --question "What objects are visible in this scene?"
```

### Analyzing Results

After processing, you can analyze the results using pandas:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_json("results/results.jsonl", lines=True)

# Filter by specific video if needed
video_df = df[df["video_url"] == "https://example.com/video.mp4"]

# Show basic stats
print(f"Total frames: {len(df)}")
print(f"Average processing time: {df['processing_times_ms'].apply(lambda x: x['total_worker_frame_time']).mean():.2f} ms")

# Plot processing times by video
df['total_time'] = df['processing_times_ms'].apply(lambda x: x['total_worker_frame_time'])
df.groupby('video_url')['total_time'].mean().plot(kind='bar', title='Average Processing Time by Video')

# Extract common words from descriptions
from collections import Counter
import re

all_text = ' '.join(df['description'].tolist())
words = re.findall(r'\b\w{3,}\b', all_text.lower())
word_freq = Counter(words).most_common(20)
print("Most common words in descriptions:", word_freq) 