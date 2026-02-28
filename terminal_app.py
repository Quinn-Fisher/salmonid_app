#!/usr/bin/env python3
"""
Terminal CLI for salmonid tracking: process video(s) without a UI.

Usage: python terminal_app.py INPUT [--tracker bytetrack|botsort] [--save_video] ...
Output: per-video JSON (and optionally annotated videos) in --output_dir.
Installation and full option list: see README.md in this package.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Optional

import cv2
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from boxmot import ByteTrack, BotSort

from config import config, AppConfig
from recording_log import get_video_timestamps
from video import process_video


VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def load_model(model_dir=None, device_override=None, cfg: Optional[AppConfig] = None):
    cfg = cfg or config
    if device_override is not None:
        device = device_override
    else:
        device = f"cuda:{cfg.cuda_device}" if torch.cuda.is_available() else "cpu"

    if model_dir is not None:
        checkpoint = model_dir
        model = AutoModelForObjectDetection.from_pretrained(
            checkpoint, local_files_only=True
        ).to(device).eval()
        image_processor = AutoImageProcessor.from_pretrained(
            checkpoint, local_files_only=True
        )
    else:
        checkpoint = cfg.model_checkpoint
        model = AutoModelForObjectDetection.from_pretrained(checkpoint).to(device).eval()
        image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    return model, image_processor, device


def build_tracker(tracker_type: str, frame_rate: int, device: str, cfg: Optional[AppConfig] = None):
    cfg = cfg or config
    if tracker_type == "bytetrack":
        return ByteTrack(
            min_conf=cfg.bytetrack_min_conf,
            track_thresh=cfg.bytetrack_track_thresh,
            match_thresh=cfg.bytetrack_match_thresh,
            track_buffer=cfg.bytetrack_track_buffer,
            frame_rate=frame_rate,
        )
    if tracker_type == "botsort":
        weights_path = Path(cfg.botsort_weights)
        if not weights_path.is_absolute():
            weights_path = Path(__file__).resolve().parent / cfg.botsort_weights
        return BotSort(
            reid_weights=weights_path,
            device=torch.device(device),
            track_high_thresh=cfg.botsort_track_high_thresh,
            track_low_thresh=cfg.botsort_track_low_thresh,
            new_track_thresh=cfg.botsort_new_track_thresh,
            track_buffer=cfg.botsort_track_buffer,
            match_thresh=cfg.botsort_match_thresh,
            half=cfg.botsort_half,
            frame_rate=frame_rate,
        )
    raise ValueError(f"Unknown tracker type: {tracker_type}")


def gather_videos(input_path):
    if os.path.isdir(input_path):
        return [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith(VIDEO_EXTS)
        ]
    return [input_path]


def main():
    parser = argparse.ArgumentParser(
        description="Run fish tracking/counting on video(s) without a UI."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to a video file or a directory containing videos.",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default=config.tracker,
        choices=["bytetrack", "botsort"],
        help="Tracker to use.",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save annotated video output.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=config.output_dir,
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Path to local model directory (optional).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (e.g., cpu, cuda:0).",
    )
    parser.add_argument(
        "--box_score_thresh",
        type=float,
        default=config.box_score_thresh,
        help="Detection score threshold.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Optional path to a recording log file (lines like 'YYYY-MM-DD HH:MM:SS - Video recording written to hard disk <name>.mp4'). Used to set video_recording_time in the output JSON. If multiple videos share the same filename, timestamps are omitted.",
    )
    args = parser.parse_args()

    model, image_processor, device = load_model(
        model_dir=args.model_dir, device_override=args.device
    )
    os.makedirs(args.output_dir, exist_ok=True)

    video_files = gather_videos(args.input)
    if not video_files:
        raise SystemExit(f"No video files found in {args.input}")

    timestamps = get_video_timestamps(video_files, args.log_file)

    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: could not open video {video_path}")
            continue
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        tracker = build_tracker(args.tracker, frame_rate, device)
        counts = process_video(
            video_path,
            model=model,
            image_processor=image_processor,
            tracker=tracker,
            save_video=args.save_video,
            output_dir=args.output_dir,
            device=device,
            box_score_thresh=args.box_score_thresh,
            font_path=config.font_path,
            min_frames_for_track=config.min_frames_for_track,
            min_displacement_frac=config.min_displacement_frac,
            min_displacement_px=config.min_displacement_px,
        )

        counts["video_recording_time"] = (
            timestamps.get(video_path, "") if timestamps is not None else ""
        )
        print(f"Counts for {video_path}:\n{json.dumps(counts, indent=2)}")
        output_json = os.path.splitext(os.path.basename(video_path))[0] + "_count.json"
        output_json = os.path.join(args.output_dir, output_json)
        with open(output_json, "w") as handle:
            json.dump(counts, handle, indent=2)
        print(f"Saved counts to {output_json}")


if __name__ == "__main__":
    main()
