"""
Gradio web UI for salmonid tracking.

Upload videos (or a zip / directory path), choose tracker and optional
recording log; optionally set a default model path and tune tracker/detection
settings. Output: JSON counts and optional annotated video downloads.
Installation and output format: see README.md in this package.
"""
import gradio as gr
import tempfile
import os
import json
import torch
import shutil
import zipfile
import cv2
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from boxmot import ByteTrack, BotSort
from video import process_video
from recording_log import get_video_timestamps
from config import config

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

SAVED_MODEL_PATH_FILE = Path(__file__).resolve().parent / "saved_model_path.txt"


def load_saved_model_path() -> str:
    """Load the persisted model directory path; return empty string if missing or unreadable."""
    try:
        if SAVED_MODEL_PATH_FILE.exists():
            return SAVED_MODEL_PATH_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    return ""


def save_saved_model_path(path: str) -> None:
    """Persist the model directory path (or empty to clear)."""
    path = (path or "").strip()
    try:
        SAVED_MODEL_PATH_FILE.write_text(path, encoding="utf-8")
    except OSError:
        pass


def clear_saved_model_path() -> None:
    """Clear the persisted model path."""
    save_saved_model_path("")


def _normalize_file_input(input_file):
    """Return a list of (path, name) from single/multiple file input."""
    if input_file is None:
        return []
    if not isinstance(input_file, list):
        input_file = [input_file]
    out = []
    for f in input_file:
        if f is None:
            continue
        if isinstance(f, dict):
            path = f.get("path")
            name = f.get("name") or (os.path.basename(path) if path else "")
        else:
            path, name = str(f), os.path.basename(str(f))
        if path:
            out.append((path, name))
    return out


def _gather_videos_from_path(path):
    """Return list of video paths from a directory or a single file."""
    if os.path.isdir(path):
        return [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith(VIDEO_EXTS)
        ]
    if path.lower().endswith(VIDEO_EXTS):
        return [path]
    return []


def load_model(model_dir=None):
    if model_dir is not None:
        checkpoint = model_dir
        device = f"cuda:{config.cuda_device}" if torch.cuda.is_available() else "cpu"
        model = AutoModelForObjectDetection.from_pretrained(checkpoint, local_files_only=True).to(device).eval()
        image_processor = AutoImageProcessor.from_pretrained(checkpoint, local_files_only=True)
    else:
        checkpoint = config.model_checkpoint
        device = f"cuda:{config.cuda_device}" if torch.cuda.is_available() else "cpu"
        model = AutoModelForObjectDetection.from_pretrained(checkpoint).to(device).eval()
        image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    return model, image_processor, device

# Default model/processor/device
model, image_processor, device = load_model()

def evaluate_video_or_zip(
    input_file,
    tracker_type,
    save_annotated_video=True,
    custom_model_zip=None,
    recording_log_file=None,
    directory_path=None,
    model_directory_path="",
    box_score_thresh=None,
    bytetrack_min_conf=None,
    bytetrack_track_thresh=None,
    bytetrack_match_thresh=None,
    bytetrack_track_buffer=None,
    botsort_track_high_thresh=None,
    botsort_track_low_thresh=None,
    botsort_new_track_thresh=None,
    botsort_track_buffer=None,
    botsort_match_thresh=None,
    botsort_half=None,
    progress=gr.Progress(),
):
    if box_score_thresh is None:
        box_score_thresh = config.box_score_thresh
    if bytetrack_min_conf is None:
        bytetrack_min_conf = config.bytetrack_min_conf
    if bytetrack_track_thresh is None:
        bytetrack_track_thresh = config.bytetrack_track_thresh
    if bytetrack_match_thresh is None:
        bytetrack_match_thresh = config.bytetrack_match_thresh
    if bytetrack_track_buffer is None:
        bytetrack_track_buffer = config.bytetrack_track_buffer
    if botsort_track_high_thresh is None:
        botsort_track_high_thresh = config.botsort_track_high_thresh
    if botsort_track_low_thresh is None:
        botsort_track_low_thresh = config.botsort_track_low_thresh
    if botsort_new_track_thresh is None:
        botsort_new_track_thresh = config.botsort_new_track_thresh
    if botsort_track_buffer is None:
        botsort_track_buffer = config.botsort_track_buffer
    if botsort_match_thresh is None:
        botsort_match_thresh = config.botsort_match_thresh
    if botsort_half is None:
        botsort_half = config.botsort_half
    results = []
    video_outputs = []
    json_outputs = []
    video_display = []
    # Resolve model: zip (this run) > saved directory path > config default
    use_custom_model = False
    custom_model_dir = None
    if custom_model_zip is not None:
        if isinstance(custom_model_zip, dict):
            custom_model_zip_path = custom_model_zip['path']
        else:
            custom_model_zip_path = custom_model_zip
        custom_model_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(custom_model_zip_path, 'r') as zip_ref:
            zip_ref.extractall(custom_model_dir)
        use_custom_model = True
        custom_model, custom_image_processor, custom_device = load_model(model_dir=custom_model_dir)
    elif model_directory_path and model_directory_path.strip() and os.path.isdir(model_directory_path.strip()):
        use_custom_model = True
        custom_model, custom_image_processor, custom_device = load_model(model_dir=model_directory_path.strip())
    else:
        custom_model = model
        custom_image_processor = image_processor
        custom_device = device

    with tempfile.TemporaryDirectory() as tmpdir:
        video_files = []
        input_basename = "videos"

        # Gather from uploaded file(s): single/multiple videos or zip(s)
        file_list = _normalize_file_input(input_file)
        for i, (path, name) in enumerate(file_list):
            if name.lower().endswith(".zip") and zipfile.is_zipfile(path):
                zip_dir = os.path.join(tmpdir, f"zip_{i}")
                os.makedirs(zip_dir, exist_ok=True)
                with zipfile.ZipFile(path, "r") as zip_ref:
                    zip_ref.extractall(zip_dir)
                for root, _, files in os.walk(zip_dir):
                    for f in files:
                        if f.startswith("._") or "__MACOSX" in root:
                            continue
                        if f.lower().endswith(VIDEO_EXTS):
                            video_files.append(os.path.join(root, f))
            elif path.lower().endswith(VIDEO_EXTS):
                dest = os.path.join(tmpdir, f"video_{i}_{name}")
                shutil.copy(path, dest)
                video_files.append(dest)
                if len(file_list) == 1 and not directory_path:
                    input_basename = os.path.splitext(name)[0]

        # Optionally add videos from a directory path (e.g. server-side path)
        if directory_path and directory_path.strip() and os.path.isdir(directory_path.strip()):
            from_dir = _gather_videos_from_path(directory_path.strip())
            video_files.extend(from_dir)
            if not file_list and from_dir:
                input_basename = os.path.splitext(os.path.basename(from_dir[0]))[0]

        if not video_files:
            return (
                "No video files found. Upload one or more videos, a zip of videos, or enter a path to a directory of videos.",
                None,
                None,
                None,
            )

        log_path = None
        if recording_log_file is not None:
            if isinstance(recording_log_file, dict):
                log_path = recording_log_file.get("path")
            else:
                log_path = recording_log_file
        timestamps = get_video_timestamps(video_files, log_path)

        total_videos = len(video_files)
        for idx, video_path in enumerate(video_files):
            progress(idx / total_videos, desc=f"Processing video {idx+1}/{total_videos} on device {device}")
            
            cap = cv2.VideoCapture(video_path)
            frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()

            # Initialize a new tracker for each video (params from UI or config defaults)
            if tracker_type == 'ByteTrack':
                tracker = ByteTrack(
                    min_conf=float(bytetrack_min_conf),
                    track_thresh=float(bytetrack_track_thresh),
                    match_thresh=float(bytetrack_match_thresh),
                    track_buffer=int(bytetrack_track_buffer),
                    frame_rate=frame_rate
                )
            elif tracker_type == 'BotSort':
                weights_path = Path(config.botsort_weights)
                if not weights_path.is_absolute():
                    weights_path = Path(__file__).resolve().parent / config.botsort_weights
                tracker = BotSort(
                    reid_weights=weights_path,
                    device=torch.device(device),
                    track_high_thresh=float(botsort_track_high_thresh),
                    track_low_thresh=float(botsort_track_low_thresh),
                    new_track_thresh=float(botsort_new_track_thresh),
                    track_buffer=int(botsort_track_buffer),
                    match_thresh=float(botsort_match_thresh),
                    half=bool(botsort_half),
                    frame_rate=frame_rate
                )

            annotated_dir = os.path.join(tmpdir, "annotated")
            os.makedirs(annotated_dir, exist_ok=True)
            counts = process_video(
                video_path,
                model=custom_model,
                image_processor=custom_image_processor,
                tracker=tracker,
                save_video=save_annotated_video,
                output_dir=annotated_dir,
                device=custom_device,
                box_score_thresh=float(box_score_thresh),
                font_path=config.font_path,
                min_frames_for_track=config.min_frames_for_track,
                min_displacement_frac=config.min_displacement_frac,
                min_displacement_px=config.min_displacement_px,
            )
            counts["video_recording_time"] = (
                timestamps.get(video_path, "") if timestamps is not None else ""
            )
            result_json = json.dumps(counts, indent=2)
            json_path = os.path.splitext(os.path.basename(video_path))[0] + "_count.json"
            json_full_path = os.path.join(annotated_dir, json_path)
            with open(json_full_path, "w") as f:
                f.write(result_json)
            annotated_video_path = os.path.join(
                annotated_dir,
                os.path.splitext(os.path.basename(video_path))[0] + "_annotated.mp4"
            )
            if save_annotated_video:
                if not os.path.exists(annotated_video_path):
                    print(f"Warning: Annotated video not found for {video_path}, skipping.")
                    continue
                video_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                shutil.copy(annotated_video_path, video_tmp.name)
                video_display.append(video_tmp.name)
                video_outputs.append((video_tmp.name, os.path.basename(annotated_video_path)))
            json_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w")
            json_tmp.write(result_json)
            json_tmp.close()
            json_outputs.append((json_tmp.name, json_path))
            results.append({
                "video": os.path.basename(video_path),
                "counts": counts
            })
        progress(1.0, desc="Done")

    if save_annotated_video and not video_display:
        return (
            "No annotated videos were created. Please check your input video(s) or try enabling 'Save and display annotated video(s)'.",
            None,
            None,
            None
        )
    if len(video_display) == 1:
        # For a single video, create a zip with both the annotated video and the JSON
        single_zip_path = tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name
        with zipfile.ZipFile(single_zip_path, 'w') as zipf:
            zipf.write(video_outputs[0][0], arcname=video_outputs[0][1])
            zipf.write(json_outputs[0][0], arcname=json_outputs[0][1])
        # Name the zip and json after the uploaded video
        zip_download_name = f"{input_basename}_count.zip"
        json_download_name = f"{input_basename}_count.json"
        # Copy the JSON to a new file with the correct name for download
        json_download_path = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
        shutil.copy(json_outputs[0][0], json_download_path)
        os.rename(json_download_path, os.path.join(os.path.dirname(json_download_path), json_download_name))
        json_download_path = os.path.join(os.path.dirname(json_download_path), json_download_name)
        return (
            json.dumps(results[0]["counts"], indent=2),
            gr.File(single_zip_path, label="Download Annotated Video and JSON", value=zip_download_name),
            gr.File(json_download_path, label="Download Results as JSON"),
            video_display
        )
    else:
        # For multiple videos, name the zip after the uploaded zip file
        zip_path = tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name
        zip_download_name = f"{input_basename}_counts.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for v, vname in video_outputs:
                zipf.write(v, arcname=vname)
            for j, jname in json_outputs:
                zipf.write(j, arcname=jname)
        return (
            json.dumps(results, indent=2),
            gr.File(zip_path, label="Download All Annotated Videos and JSONs (zip)", value=zip_download_name),
            None,
            video_display
        )

DESCRIPTION = """
# Salmonid Tracking Video Evaluation
Upload one or more video files, or a zip of videos, or enter a path to a directory of videos. Set a default model directory path below (optional) and click Save as default so it is remembered next time. You can still upload a custom model zip for a one-off run. The results will be shown as JSON and annotated video(s) will be displayed and available for download.
"""


def _run_evaluate(
    model_dir_path,
    input_file,
    tracker_type,
    save_annotated_video,
    custom_model_zip,
    recording_log_file,
    directory_path,
    box_score_thresh,
    bytetrack_min_conf,
    bytetrack_track_thresh,
    bytetrack_match_thresh,
    bytetrack_track_buffer,
    botsort_track_high_thresh,
    botsort_track_low_thresh,
    botsort_new_track_thresh,
    botsort_track_buffer,
    botsort_match_thresh,
    botsort_half,
    progress=gr.Progress(),
):
    """Wrapper so model_directory_path and tracker/detection settings are passed from UI."""
    return evaluate_video_or_zip(
        input_file,
        tracker_type,
        save_annotated_video=save_annotated_video,
        custom_model_zip=custom_model_zip,
        recording_log_file=recording_log_file,
        directory_path=directory_path,
        model_directory_path=model_dir_path or "",
        box_score_thresh=box_score_thresh,
        bytetrack_min_conf=bytetrack_min_conf,
        bytetrack_track_thresh=bytetrack_track_thresh,
        bytetrack_match_thresh=bytetrack_match_thresh,
        bytetrack_track_buffer=bytetrack_track_buffer,
        botsort_track_high_thresh=botsort_track_high_thresh,
        botsort_track_low_thresh=botsort_track_low_thresh,
        botsort_new_track_thresh=botsort_new_track_thresh,
        botsort_track_buffer=botsort_track_buffer,
        botsort_match_thresh=botsort_match_thresh,
        botsort_half=botsort_half,
        progress=progress,
    )


def _save_default(path):
    save_saved_model_path(path or "")
    return "Saved as default."


def _clear_default():
    clear_saved_model_path()
    return "", "Cleared."


with gr.Blocks(title="Salmonid Tracking Video Evaluation") as iface:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        model_path_tb = gr.Textbox(
            label="Model directory path (optional)",
            placeholder="/path/to/model",
            value=load_saved_model_path(),
        )
        save_btn = gr.Button("Save as default")
        clear_btn = gr.Button("Clear default")
    model_status_md = gr.Markdown(visible=True)

    video_file = gr.File(label="Upload Video(s) or Zip", file_count="multiple", type="filepath")
    tracker_type = gr.Dropdown(["ByteTrack", "BotSort"], label="Tracker", value="ByteTrack")
    save_annotated_video = gr.Checkbox(label="Save and display annotated video(s)", value=True)
    custom_model_zip = gr.File(label="Upload Custom Model Zip (optional)", file_count="single", type="filepath")
    recording_log_file = gr.File(label="Upload recording log (optional)", file_count="single", type="filepath")
    directory_path = gr.Textbox(label="Or path to directory of videos (optional)", placeholder="/path/to/videos", value="")

    with gr.Accordion("Tracker & detection settings", open=False):
        box_score_thresh_in = gr.Number(
            value=config.box_score_thresh,
            label="Detection confidence threshold",
            minimum=0.0,
            maximum=1.0,
            step=0.05,
        )
        gr.Markdown("**ByteTrack**")
        bytetrack_min_conf_in = gr.Number(value=config.bytetrack_min_conf, label="min_conf", minimum=0.0, maximum=1.0, step=0.01)
        bytetrack_track_thresh_in = gr.Number(value=config.bytetrack_track_thresh, label="track_thresh", minimum=0.0, maximum=1.0, step=0.01)
        bytetrack_match_thresh_in = gr.Number(value=config.bytetrack_match_thresh, label="match_thresh", minimum=0.0, maximum=1.0, step=0.01)
        bytetrack_track_buffer_in = gr.Number(value=config.bytetrack_track_buffer, label="track_buffer", minimum=1, maximum=300, step=1, precision=0)
        gr.Markdown("**BotSort**")
        botsort_track_high_thresh_in = gr.Number(value=config.botsort_track_high_thresh, label="track_high_thresh", minimum=0.0, maximum=1.0, step=0.01)
        botsort_track_low_thresh_in = gr.Number(value=config.botsort_track_low_thresh, label="track_low_thresh", minimum=0.0, maximum=1.0, step=0.01)
        botsort_new_track_thresh_in = gr.Number(value=config.botsort_new_track_thresh, label="new_track_thresh", minimum=0.0, maximum=1.0, step=0.01)
        botsort_track_buffer_in = gr.Number(value=config.botsort_track_buffer, label="track_buffer", minimum=1, maximum=300, step=1, precision=0)
        botsort_match_thresh_in = gr.Number(value=config.botsort_match_thresh, label="match_thresh", minimum=0.0, maximum=1.0, step=0.01)
        botsort_half_in = gr.Checkbox(value=config.botsort_half, label="half (FP16 ReID)")

    run_btn = gr.Button("Run", variant="primary")

    out_json = gr.Textbox(label="Tracking Results (JSON)")
    out_download_zip = gr.File(label="Download Annotated Video(s) and JSON(s)")
    out_download_json = gr.File(label="Download Results as JSON (single video only)", visible=False)
    out_gallery = gr.Gallery(label="Annotated Video Gallery", type="video")

    run_btn.click(
        fn=_run_evaluate,
        inputs=[
            model_path_tb,
            video_file,
            tracker_type,
            save_annotated_video,
            custom_model_zip,
            recording_log_file,
            directory_path,
            box_score_thresh_in,
            bytetrack_min_conf_in,
            bytetrack_track_thresh_in,
            bytetrack_match_thresh_in,
            bytetrack_track_buffer_in,
            botsort_track_high_thresh_in,
            botsort_track_low_thresh_in,
            botsort_new_track_thresh_in,
            botsort_track_buffer_in,
            botsort_match_thresh_in,
            botsort_half_in,
        ],
        outputs=[out_json, out_download_zip, out_download_json, out_gallery],
    )

    save_btn.click(
        fn=_save_default,
        inputs=[model_path_tb],
        outputs=[model_status_md],
    )

    clear_btn.click(
        fn=_clear_default,
        inputs=[],
        outputs=[model_path_tb, model_status_md],
    )

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860) 
