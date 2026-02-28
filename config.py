"""
Default configuration for the Gradio and terminal apps.

Model checkpoint, detection threshold, tracker choice (ByteTrack/BotSort), and
tracker parameters. Edit AppConfig defaults to change behavior; see README.md.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Configuration for inference and tracking. No training parameters.
# Edit these values to change default behavior across the Gradio app and terminal script.
@dataclass
class AppConfig:

    # ----- Model (inference only) -----
    # Default model when no custom model is provided. HuggingFace id or path to local checkpoint.
    model_checkpoint: str = "PekingU/rtdetr_v2_r50vd"
    # CUDA device index, e.g. '0' or '1'. Ignored if CUDA not available.
    cuda_device: str = "0"

    # ----- Detection -----
    # Detection score threshold. Detections below this are discarded.
    box_score_thresh: float = 0.6

    # ----- Tracker choice -----
    # Tracker to use: 'bytetrack' or 'botsort'.
    tracker: str = "bytetrack"

    # ----- ByteTrack parameters -----
    bytetrack_min_conf: float = 0.6
    bytetrack_track_thresh: float = 0.12
    bytetrack_match_thresh: float = 0.99
    bytetrack_track_buffer: int = 30

    # ----- BotSort parameters -----
    # Path to ReID weights file, relative to repo root or absolute.
    botsort_weights: str = "botsort_weights/osnet_x0_25_msmt17.pt"
    botsort_track_high_thresh: float = 0.6
    botsort_track_low_thresh: float = 0.3
    botsort_new_track_thresh: float = 0.7
    botsort_track_buffer: int = 60
    botsort_match_thresh: float = 0.8
    # Use half precision for ReID (faster, may be less accurate).
    botsort_half: bool = False

    # ----- Video / process_video -----
    # Path to font used for on-video annotations.
    font_path: str = "arial.ttf"
    # Minimum number of frames a track must have to be included in counts.
    min_frames_for_track: int = 5
    # Minimum horizontal movement (fraction of frame width) to assign Left/Right direction.
    min_displacement_frac: float = 0.05
    # Minimum horizontal movement in pixels (used together with min_displacement_frac; max of the two is used).
    min_displacement_px: float = 20.0

    # ----- Output (terminal script defaults) -----
    # Default directory for saving count JSON and annotated videos when using the terminal script.
    output_dir: str = "/home/quinn"

    def __post_init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_device


# Default config instance
config = AppConfig()
