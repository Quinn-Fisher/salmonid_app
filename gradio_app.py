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
from config import config

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

def evaluate_video_or_zip(input_file, tracker_type, save_annotated_video=True, custom_model_zip=None, progress=gr.Progress()):
    results = []
    video_outputs = []
    json_outputs = []
    video_display = []
    # Handle custom model zip if provided
    use_custom_model = False
    custom_model_dir = None
    if custom_model_zip is not None:
        if isinstance(custom_model_zip, dict):
            custom_model_zip_path = custom_model_zip['path']
        else:
            custom_model_zip_path = custom_model_zip
        # Extract zip to temp dir
        custom_model_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(custom_model_zip_path, 'r') as zip_ref:
            zip_ref.extractall(custom_model_dir)
        use_custom_model = True
        # Load model/processor/tracker/device from extracted dir
        custom_model, custom_image_processor, custom_device = load_model(model_dir=custom_model_dir)
    else:
        custom_model = model
        custom_image_processor = image_processor
        custom_device = device

    with tempfile.TemporaryDirectory() as tmpdir:
        if isinstance(input_file, dict):
            input_path = os.path.join(tmpdir, os.path.basename(input_file['name']))
            shutil.copy(input_file['path'], input_path)
            input_basename = os.path.splitext(os.path.basename(input_file['name']))[0]
        else:
            input_path = os.path.join(tmpdir, os.path.basename(input_file))
            shutil.copy(input_file, input_path)
            input_basename = os.path.splitext(os.path.basename(input_file))[0]
        video_files = []
        if zipfile.is_zipfile(input_path):
            with zipfile.ZipFile(input_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
                for root, _, files in os.walk(tmpdir):
                    for f in files:
                        # Ignore macOS resource fork files and __MACOSX directory
                        if f.startswith("._") or "__MACOSX" in root:
                            continue
                        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                            video_files.append(os.path.join(root, f))
        else:
            video_files = [input_path]

        total_videos = len(video_files)
        for idx, video_path in enumerate(video_files):
            progress(idx / total_videos, desc=f"Processing video {idx+1}/{total_videos} on device {device}")
            
            cap = cv2.VideoCapture(video_path)
            frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()

            # Initialize a new tracker for each video (params from config)
            if tracker_type == 'ByteTrack':
                tracker = ByteTrack(
                    min_conf=config.bytetrack_min_conf,
                    track_thresh=config.bytetrack_track_thresh,
                    match_thresh=config.bytetrack_match_thresh,
                    track_buffer=config.bytetrack_track_buffer,
                    frame_rate=frame_rate
                )
            elif tracker_type == 'BotSort':
                weights_path = Path(config.botsort_weights)
                if not weights_path.is_absolute():
                    weights_path = Path(__file__).resolve().parent / config.botsort_weights
                tracker = BotSort(
                    reid_weights=weights_path,
                    device=torch.device(device),
                    track_high_thresh=config.botsort_track_high_thresh,
                    track_low_thresh=config.botsort_track_low_thresh,
                    new_track_thresh=config.botsort_new_track_thresh,
                    track_buffer=config.botsort_track_buffer,
                    match_thresh=config.botsort_match_thresh,
                    half=config.botsort_half,
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
                box_score_thresh=config.box_score_thresh,
                font_path=config.font_path,
                min_frames_for_track=config.min_frames_for_track,
                min_displacement_frac=config.min_displacement_frac,
                min_displacement_px=config.min_displacement_px,
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

description = """
# Salmonid Tracking Video Evaluation
Upload a video file or a zip file containing videos to run fish tracking and counting. Optionally, upload a zip containing a custom model (config.json, preprocessor_config.json, model.safetensors) to use for inference. The results will be shown as JSON and annotated video(s) will be displayed and available for download.
"""

iface = gr.Interface(
    fn=evaluate_video_or_zip,
    inputs=[
        gr.File(label="Upload Video or Zip of Videos"),
        gr.Dropdown(['ByteTrack', 'BotSort'], label="Tracker", value='ByteTrack'),
        gr.Checkbox(label="Save and display annotated video(s)", value=True),
        gr.File(label="Upload Custom Model Zip (optional)", file_count="single", type="filepath")
    ],
    outputs=[
        gr.Textbox(label="Tracking Results (JSON)"),
        gr.File(label="Download Annotated Video(s) and JSON(s)"),
        gr.File(label="Download Results as JSON (single video only)", visible=False),
        gr.Gallery(label="Annotated Video Gallery", visible=True, type="video")
    ],
    title="Salmonid Tracking Video Evaluation",
    description=description,
    allow_flagging="never"
)

# Postprocess to show either single video or gallery
iface.postprocess = lambda outputs: (
    outputs[0],
    outputs[1],
    outputs[2],
    outputs[3]
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860) 
