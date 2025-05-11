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

## Output

Results are saved in two formats:

1. `<video_name>.jsonl` - Each line contains a JSON object with data for a single frame. Results from multiple jobs processing the same video are appended to this file, allowing for accumulated results over time.

2. `results/full_results/<video_name>_<job_id>.json` - Complete job data including metadata and processing statistics, stored in a separate "full_results" directory to keep the main results directory clean.

The JSONL format is particularly useful for data analysis, as it can be easily loaded into pandas or other data processing tools:

```python
import pandas as pd

# Load data from JSONL file
df = pd.read_json("results/video_name.jsonl", lines=True)

# Show frame descriptions
print(df[["timestamp_ms", "description"]])

# Filter by specific job ID if needed
job_results = df[df["job_id"] == "job_12345678"]
```

This approach allows you to:
- Maintain a single file per video, regardless of how many times you process it
- Easily accumulate results from multiple processing runs (with different parameters)
- Still track which results came from which job via the embedded job_id 

## Performance

Based on our tests, here are the actual processing times with warm containers (no cold start):

| Video                                  | Duration | Frames Processed | Processing Time | FPS Setting |
|----------------------------------------|----------|------------------|----------------|------------|
| 214409_tiny.mp4 (cloud timelapse)      | 50 sec   | 251 frames       | 18.26 seconds  | 5 fps      |
| Big_Buck_Bunny_360_10s_1MB.mp4         | 10 sec   | 50 frames        | 4.27 seconds   | 5 fps      |
| ForBiggerBlazes.mp4                    | 15 sec   | 72 frames        | 5.68 seconds   | 5 fps      |

For comparison, our earlier test with cold start and 1 FPS:

| Video                                  | Duration | Frames Processed | Processing Time | Notes |
|----------------------------------------|----------|------------------|----------------|-------|
| 214409_tiny.mp4 (cloud timelapse)      | 50 sec   | 51 frames        | 38.87 seconds  | Includes ~30s cold start time; actual processing time was ~9s |

Processing time depends on:
- Cold start time (first request after deployment can take 20-30s)
- Number of frames extracted (based on FPS setting)
- Model inference time per frame (~1.5s per frame with Moondream2)
- Current load on the Modal infrastructure
- Video complexity
- Parallel processing (with 30 max GPU workers, processing scales well with more frames)

Keep in mind that the `min_containers=1` parameter in the deployment helps reduce cold starts for subsequent requests by keeping at least one worker warm.

## Examples

### Single Video Processing

```bash
# Process a single video with default parameters
python moondream_client.py --video "https://cdn.pixabay.com/video/2024/05/29/214409_tiny.mp4"

# Process with custom FPS and question
python moondream_client.py --video "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4" --fps 2 --question "What animals are visible in this scene?"

# Save results to a custom directory
python moondream_client.py --video "https://example.com/my_video.mp4" --output-dir "my_results"
```

### Batch Processing

```bash
# Create a file with video URLs
echo "https://cdn.pixabay.com/video/2024/05/29/214409_tiny.mp4
https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4
https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4" > videos_to_process.txt

# Process all videos in batch
python moondream_client.py --batch videos_to_process.txt --fps 1 --question "What objects are visible in this scene?"
```

### Analyzing Results

After processing, you can analyze the results using pandas:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_json("results/214409_tiny.mp4.jsonl", lines=True)

# Show basic stats
print(f"Total frames: {len(df)}")
print(f"Average processing time: {df['processing_times_ms'].apply(lambda x: x['total_worker_frame_time']).mean():.2f} ms")

# Plot processing times
df['total_time'] = df['processing_times_ms'].apply(lambda x: x['total_worker_frame_time'])
df.plot(x='timestamp_ms', y='total_time', title='Processing Time by Frame')

# Extract common words from descriptions
from collections import Counter
import re

all_text = ' '.join(df['description'].tolist())
words = re.findall(r'\b\w{3,}\b', all_text.lower())
word_freq = Counter(words).most_common(20)
print("Most common words in descriptions:", word_freq) 