import logging
import time
from typing import Any, Dict, List
from cog import BasePredictor, Input
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from .video_analyzer import VideoAnalyzer

logging.basicConfig(level=logging.INFO)

MODEL_NAME = "vikhyatk/moondream2"
REVISION = "2025-04-14"
MODEL_CACHE = "checkpoints"

class Predictor(BasePredictor):
    def setup(self, num_models: int = 3):
        """
        Load Moondream models
        
        Args:
            num_models: Number of model instances to load (default: 3)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        
        # Validate num_models
        if num_models < 1:
            logging.warning(f"Invalid num_models: {num_models}. Setting to 1.")
            num_models = 1
        
        # Store the number of models
        self.num_models = num_models
        logging.info(f"Loading {num_models} Moondream model instances")
        
        # Load models in a list
        self.models = []
        for i in range(num_models):
            logging.info(f"Loading Moondream model {i+1}")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                cache_dir=MODEL_CACHE,
                revision=REVISION
            )
            model = model.to(self.device)
            model.eval()
            self.models.append(model)
            logging.info(f"Moondream model {i+1} loaded successfully")
        
        # For backward compatibility, also store individual models
        if num_models >= 1:
            self.model_moondream1 = self.models[0]
        if num_models >= 2:
            self.model_moondream2 = self.models[1]
        if num_models >= 3:
            self.model_moondream3 = self.models[2]
        
        logging.info(f"Models are on device: {next(self.models[0].parameters()).device}")

        # Load tokenizer (only need one)
        logging.info("Loading Moondream tokenizer")
        self.tokenizer_moondream = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            revision=REVISION,
            cache_dir=MODEL_CACHE
        )
        logging.info("Moondream tokenizer loaded successfully")
        
        # Create video analyzer
        self.video_analyzer = VideoAnalyzer(self.models, self.tokenizer_moondream)

    def predict(
        self, 
        video_url: str = Input(description="URL of the video to analyse"),
        debug_log: bool = Input(description="Enable detailed timing logs for performance analysis", default=False),
        use_queue: bool = Input(description="Use queue-based processing (for distributed processing)", default=True)
    ) -> Any:
        """Analyze the video and return the results"""
        start_time = time.time()
        
        # Use the video analyzer
        results, timing_metrics = self.video_analyzer.analyze_video(
            video_url, 
            debug_log=debug_log,
            use_queue=use_queue
        )
        
        # Add model loading time to timing metrics
        timing_metrics["Model loading time (seconds)"] = 0  # This would need to be captured from setup
        
        # Return both results and timing metrics
        return {
            "results": results,
            "timing_metrics": timing_metrics
        } 