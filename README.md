# Salmonid Tracking (Gradio & Terminal)

Object detection and multi-object tracking for salmonid fish in video. Runs an RT-DETR-style model per frame, links detections across frames with ByteTrack or BotSort, and outputs per-fish counts with direction, side, and class scores (including a Background class). Supports optional recording logs for video timestamps and a Gradio UI or a CLI.

---

## Installation (from scratch, with NVIDIA GPU)

This section is for a machine that has an NVIDIA GPU but does not yet have Python or CUDA tooling installed. PyTorch includes its own CUDA runtime, so you only need a working **NVIDIA driver**; you do not need to install the CUDA Toolkit separately.

### 1. Install Python 3.10 or 3.11

- **Windows:** Download the installer from [python.org/downloads](https://www.python.org/downloads/). Run it and check **“Add Python to PATH”**. Open a new Command Prompt or PowerShell after installing.
- **Linux (Debian/Ubuntu):**  
  `sudo apt update && sudo apt install python3.11 python3.11-venv python3-pip`
- **macOS:**  
  `brew install python@3.11`  
  (or download from python.org).

Check that it works:

```bash
python --version
# or: python3 --version
```

You should see something like `Python 3.11.x`.

### 2. Install or update the NVIDIA driver

The GPU code needs an NVIDIA driver that supports the CUDA version used by PyTorch (e.g. CUDA 11.8 or 12.1). Newer drivers support older CUDA runtimes.

- **Windows:** Download the latest Game Ready or Studio driver from [NVIDIA Driver Download](https://www.nvidia.com/Download/index.aspx). Install and reboot if asked.
- **Linux:** Install the proprietary driver (e.g. `sudo apt install nvidia-driver-535` or use “Additional Drivers” in Software & Updates). Reboot, then run `nvidia-smi` in a terminal; you should see your GPU and driver version.

Keep `nvidia-smi` output handy; the “CUDA Version” shown there is the maximum your driver supports (PyTorch will use a compatible bundled runtime).

### 3. Open a terminal in the project folder

If you cloned a repo or copied the app somewhere:

```bash
cd /path/to/salmonid_gradio_app
```

(Use your actual path.)

### 4. Create and activate a virtual environment

Using the Python you installed:

```bash
python -m venv venv
```

Then activate it:

- **Windows (Command Prompt):**  
  `venv\Scripts\activate.bat`
- **Windows (PowerShell):**  
  `venv\Scripts\Activate.ps1`
- **Linux/macOS:**  
  `source venv/bin/activate`

Your prompt should show `(venv)`.

### 5. Install PyTorch with GPU support, then the rest

Install PyTorch and torchvision built for CUDA first. Choose one of these (CUDA 11.8 is widely compatible; use 12.1 if your driver is recent):

**CUDA 11.8:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1 (newer drivers):**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

If you are unsure, start with `cu118`. Then install the project dependencies:

```bash
pip install -r requirements.txt
```

### 6. Check that the GPU is seen

With the venv still active:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

You want `CUDA available: True` and your GPU name. If you see `False`, the driver may be too old, or PyTorch was installed without CUDA (re-run the `pip install torch torchvision --index-url ...` step for your OS and Python from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)).

### 7. Optional: BotSort ReID weights

If you use the **BotSort** tracker, the app expects ReID weights at `botsort_weights/osnet_x0_25_msmt17.pt` (see `config.py`). Download that file if missing (e.g. from boxmot docs or the repo they point to). **ByteTrack** does not need extra weights.

---

## Installation without a GPU (CPU only)

If you have no NVIDIA GPU or prefer to run on CPU (slower):

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

The default `requirements.txt` pulls in a CPU build of PyTorch. Inference will use the CPU.

---

## Usage

### Gradio app (web UI)

Start the server:

```bash
python gradio_app.py
```

By default it listens on `http://0.0.0.0:7860`. Open that URL in a browser.

**Inputs:**

- **Model directory path (optional)**  
  Server-side path to a folder with a HuggingFace-style object detection model (`config.json`, `preprocessor_config.json`, `model.safetensors`). Use **Save as default** to remember it; **Clear default** to go back to the built-in default (e.g. PekingU/rtdetr_v2_r50vd).

- **Upload Video(s) or Zip**  
  One or more video files (e.g. `.mp4`), or a single zip of videos. No need to zip if you can upload multiple files.

- **Tracker**  
  ByteTrack (default) or BotSort.

- **Save and display annotated video(s)**  
  If checked, outputs include annotated videos with bounding boxes and labels.

- **Upload Custom Model Zip (optional)**  
  One-off run with a model from a zip; overrides the default/saved model path for that run only.

- **Upload recording log (optional)**  
  Log file with lines like `YYYY-MM-DD HH:MM:SS.mmm - Video recording written to hard disk 1.mp4`. Used to set `video_recording_time` in the output JSON by matching video filenames. If multiple videos share the same filename, timestamps are not applied.

- **Or path to directory of videos (optional)**  
  Server path to a folder of videos; processed together with any uploaded files.

- **Tracker & detection settings** (accordion)  
  Detection confidence threshold; ByteTrack/BotSort parameters. Defaults come from `config.py`; change as needed.

Click **Run** to process. Results appear as JSON and (optionally) annotated videos and downloads.

### Terminal app (CLI)

Process one video or a directory of videos without a UI:

```bash
python terminal_app.py INPUT [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|--------------|
| `input` | Path to a single video file or a directory containing videos (`.mp4`, `.avi`, `.mov`, `.mkv`). |
| `--tracker` | `bytetrack` (default) or `botsort`. |
| `--save_video` | Save annotated videos (e.g. `*_annotated.mp4`) in the output directory. |
| `--output_dir` | Where to write JSON and (if `--save_video`) annotated videos. Default from `config.py`. |
| `--model_dir` | Path to a local model directory (optional); overrides default checkpoint. |
| `--device` | e.g. `cpu` or `cuda:0`. |
| `--box_score_thresh` | Detection confidence threshold (default from config). |
| `--log_file` | Optional recording log path; sets `video_recording_time` in JSON (see recording log format below). |

**Examples:**

```bash
# Single video, default model and ByteTrack
python terminal_app.py /path/to/video.mp4

# Directory of videos, save annotated outputs, custom model
python terminal_app.py /path/to/videos/ --save_video --model_dir /path/to/my_model --output_dir ./out

# With recording log for timestamps
python terminal_app.py /path/to/videos/ --log_file /path/to/Sc1_Oct31_to_Nov28_2025.log
```

Per-video count JSON is written as `<basename>_count.json` in the output directory (and printed to stdout).

---

## Output

### JSON structure

Each video produces one JSON object. Top-level keys:

- **`video_recording_time`** (string)  
  Timestamp when the video was written, if a recording log was provided and the filename matched uniquely; otherwise `""`.

- **`"1"`, `"2"`, …** (per-track objects)  
  One entry per tracked fish (track ID as string). Each track has:

| Field | Type | Description |
|-------|------|-------------|
| `video_path` | string | Path to the source video. |
| `direction` | string | `"Left"`, `"Right"`, or `"Unknown"` from horizontal movement. |
| `first_side` | string | `"Left"` or `"Right"` (or `"None"`) at first detection. |
| `last_side` | string | Same at last detection. |
| `entered_frame` | string | `"True"` if the fish was not seen at the very start of the video. |
| `exited_frame` | string | `"True"` if the fish was not seen at the very end. |
| `class_scores` | object | Mean class probabilities over the track: one key per class (e.g. `Chinook`, `Coho`, `Atlantic`, `Rainbow Trout`, `Brown Trout`, `Background`). Probabilities sum to 1; `Background` is the no-object class. |
| `top_class` | string | Class with the highest mean probability (can be `"Background"` if uncertain). |

Tracks are only included if they span at least `min_frames_for_track` frames (see `config.py`).

**Example (excerpt):**

```json
{
  "video_recording_time": "2025-10-31 16:42:09.357",
  "1": {
    "video_path": "/path/to/1.mp4",
    "direction": "Right",
    "first_side": "Left",
    "last_side": "Right",
    "entered_frame": "False",
    "exited_frame": "True",
    "class_scores": {
      "Chinook": 0.72,
      "Coho": 0.05,
      "Atlantic": 0.02,
      "Rainbow Trout": 0.01,
      "Brown Trout": 0.01,
      "Background": 0.19
    },
    "top_class": "Chinook"
  }
}
```

### Recording log format

Optional. Used to set `video_recording_time` in the JSON by matching **video filename** (no path).

- Lines of the form:  
  `YYYY-MM-DD HH:MM:SS.mmm - Video recording written to hard disk <filename>`
- Example:  
  `2025-10-31 16:42:09.357 - Video recording written to hard disk 1.mp4`
- If you process multiple videos that share the same filename, the log is not used and timestamps are left blank (and a warning is printed in the terminal app).

---

## Configuration

Edit `config.py` to change default model, device, detection threshold, tracker (ByteTrack/BotSort), and tracker parameters. Those defaults are used by both the Gradio app and the terminal app unless overridden in the UI or via CLI flags.

---

## Module overview

- **`video.py`** — Frame-by-frame detection, tracking, and aggregation into per-fish counts; outputs the structure above. See README for output format.
- **`gradio_app.py`** — Gradio UI: model path, videos (or zip/directory), tracker, recording log, and optional tracker/detection settings.
- **`terminal_app.py`** — CLI entrypoint; loads model, gathers videos, calls `video.process_video`, writes JSON and optional annotated videos.
- **`recording_log.py`** — Parses recording log files and maps video filenames to timestamps.
- **`config.py`** — Shared defaults for model, detection, and tracker parameters.
