"""
Video processing: run object detection and multi-object tracking on a video.

Reads frames, runs the detector (RT-DETR-style), links detections across frames
with ByteTrack or BotSort, and returns per-fish counts with direction, side,
entered/exited flags, and class_scores (fish classes + Background). Class
probabilities use full softmax over all logits including the no-object class.

Output JSON structure and usage: see README.md in this package.
"""
import os
import sys
import cv2
import numpy as np
import torch
import torchvision
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import imageio.v2 as imageio
import logging

# Package fonts dir: drop a .ttf here and annotations use it on any OS (size still scaled by video).
_PACKAGE_FONTS_DIR = Path(__file__).resolve().parent / "fonts"


def _annotation_font_candidates(font_path_config):
    """Return (regular_paths, bold_paths) to try for annotation font. Cross-platform; only cares that size is applied."""
    regular, bold = [], []
    # 1. Package fonts dir first — one font here works on every OS
    if _PACKAGE_FONTS_DIR.is_dir():
        for p in sorted(_PACKAGE_FONTS_DIR.iterdir()):
            if p.suffix.lower() in (".ttf", ".ttc", ".otf"):
                regular.append(str(p))
        if font_path_config and not os.path.isabs(font_path_config):
            custom = _PACKAGE_FONTS_DIR / os.path.basename(font_path_config)
            if custom.is_file() and str(custom) not in regular:
                regular.insert(0, str(custom))
    # 2. OS-specific system fonts (order by likelihood)
    if sys.platform == "darwin":
        regular += [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNSText.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
        bold += [
            "/Library/Fonts/Microsoft/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        ]
    elif sys.platform == "win32":
        windir = os.environ.get("WINDIR", "C:\\Windows")
        f = os.path.join(windir, "Fonts", "arial.ttf")
        regular += [f, os.path.join(windir, "Fonts", "Arial.ttf")]
        bold += [os.path.join(windir, "Fonts", "arialbd.ttf"), os.path.join(windir, "Fonts", "Arial Bold.ttf")]
    else:
        regular += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
        ]
        bold += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        ]
    return regular, bold

# Set up logging
logging.basicConfig(filename='video_debug.log', level=logging.DEBUG, 
                    format='%(asctime)s - FRAME %(message)s', filemode='w')


id2label = {0: 'Chinook', 1: 'Coho', 2: 'Atlantic', 3: 'Rainbow Trout', 4: 'Brown Trout', 5: 'Background'}
label2id = {'Chinook': 0, 'Coho': 1, 'Atlantic': 2, 'Rainbow Trout': 3, 'Brown Trout': 4, 'Background': 5}
num_classes = len(id2label)  # 6 (5 fish + Background)


def process_video(
    video_path,
    model,
    image_processor,
    tracker,
    save_video=False,
    output_dir=None,
    device="cuda",
    box_score_thresh=0.1,
    font_path="arial.ttf",
    id2label=None,
    min_frames_for_track=5,
    min_displacement_frac=0.05,
    min_displacement_px=20.0,
):
    """
    Process a video for object detection and tracking.

    Args:
        video_path (str): Path to the video file.
        model: Object detection model.
        image_processor: Image processor for the model.
        tracker: Multi-object tracker instance.
        save_video (bool): Whether to save annotated video output.
        output_dir (str): Directory to save annotated video (if save_video=True).
        device (str): Device to run inference on.
        box_score_thresh (float): Detection score threshold.
        font_path (str): Path to font for annotation.
        id2label (dict): Mapping from class index to class name.
        min_frames_for_track (int): Minimum frames a track must have to be included in counts.
        min_displacement_frac (float): Min horizontal movement as fraction of frame width for direction.
        min_displacement_px (float): Min horizontal movement in pixels (max with frac is used).

    Returns:
        List of dicts with track info for each detected object.
    """
    if id2label is None:
        id2label = globals().get("id2label", {})

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_size = (width, height)

    # Prepare output path for imageio
    if save_video:
        if output_dir is None:
            output_path = os.path.splitext(video_path)[0] + "_annotated.mp4"
        else:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir, os.path.splitext(os.path.basename(video_path))[0] + "_annotated.mp4"
            )
        frames = []

    to_pil = torchvision.transforms.ToPILImage()

    # Load a scalable font so size is respected (Pillow's load_default() is tiny and ignores size).
    font_size = max(44, min(96, int(min(width, height) / 18)))
    regular_paths, bold_paths = _annotation_font_candidates(font_path)
    font = None
    for path in regular_paths:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except (IOError, OSError):
            continue
    if font is None:
        font = ImageFont.load_default()
    font_bold = font
    for path in bold_paths:
        try:
            font_bold = ImageFont.truetype(path, font_size)
            break
        except (IOError, OSError):
            continue

    objects_detected = {}

    for frame_idx in tqdm(range(total_frames), desc=f"Processing {os.path.basename(video_path)}"):
        ret, frame = cap.read()
        if not ret:
            break

        objects_detected['final_frame_for_video'] = frame_idx

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        if save_video:
            draw = ImageDraw.Draw(pil_image)

        processed = image_processor(pil_image, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, return_dict=True)

        detections = image_processor.post_process_object_detection(
                outputs, target_sizes=torch.tensor([[max(pil_image.size), max(pil_image.size)]]), threshold=box_score_thresh
            )[0]

        # Per-detection class probabilities (same order as post_process detections).
        # Full softmax over all logits including no-object (last dim); we keep Background in class_scores.
        probs = torch.softmax(outputs.logits, dim=-1)
        probs_full = probs[0]  # (num_queries, num_classes) including Background at last index
        scores_per_query = probs_full[:, :-1].max(dim=-1).values  # max over fish classes for thresholding
        keep = scores_per_query > box_score_thresh
        class_probs_per_detection = probs_full[keep].cpu().numpy()

        boxes = detections['boxes'].cpu().numpy()
        scores = detections['scores'].cpu().numpy()
        labels = detections['labels'].cpu().numpy()
        dets = np.column_stack((boxes, scores, labels))

        # Update tracker every frame so tracks stay alive (Kalman prediction) when detections drop out.
        res = tracker.update(dets, frame)

        # Update and draw tracked objects. Tracker output is in track order, not detection order.
        # boxmot appends original detection index as last column (det_ind); use it to link to class probs.
        for track in res:
            # Unpack: x1, y1, x2, y2, track_id, conf, cls, det_ind
            x1, y1, x2, y2, track_id, conf, cls, det_ind = track
            x1, y1, x2, y2, track_id, cls, det_ind = int(x1), int(y1), int(x2), int(y2), int(track_id), int(cls), int(det_ind)
            prob_vec = (
                class_probs_per_detection[det_ind]
                if 0 <= det_ind < len(class_probs_per_detection)
                else np.zeros(num_classes, dtype=np.float64)  # includes Background
            )

            if track_id in objects_detected.keys():
                objects_detected[track_id]['boxes'].append([x1, y1, x2, y2])
                objects_detected[track_id]['classes'].append(cls)
                objects_detected[track_id]['confs'].append(conf)
                objects_detected[track_id]['frames'].append(frame_idx)
                objects_detected[track_id]['class_probs'].append(prob_vec)
            else:
                objects_detected[track_id] = {}
                objects_detected[track_id]['boxes'] = [[x1, y1, x2, y2]]
                objects_detected[track_id]['classes'] = [cls]
                objects_detected[track_id]['confs'] = [conf]
                objects_detected[track_id]['frames'] = [frame_idx]
                objects_detected[track_id]['class_probs'] = [prob_vec]
        

        
            # if save_video:
                # draw.rectangle([(x1, y1), (x2, y2)], outline="orange", width=5)

        # Draw detection (magenta) boxes with Fish ID + class + score in bold
        if save_video:
            det_idx_to_track_id = {int(track[7]): int(track[4]) for track in res}
            for det_idx, (box, score, label) in enumerate(zip(detections['boxes'].cpu(), detections['scores'].cpu(), detections['labels'].cpu())):
                if score > box_score_thresh:
                    x1, y1, x2, y2 = map(int, box)
                    draw.rectangle([(x1, y1), (x2, y2)], outline="magenta", width=5)
                    class_name = id2label.get(int(label), str(label))
                    track_id = det_idx_to_track_id.get(det_idx, "?")
                    draw.text(
                        (x1, y1 - font_size - 5),
                        f"Fish {track_id}: {class_name} ({score:.2f})",
                        fill="magenta",
                        font=font_bold,
                    )

        if save_video:
            annotated_frame = np.array(pil_image)
            frames.append(annotated_frame)

    cap.release()
    if save_video and frames:
        with imageio.get_writer(output_path, fps=fps, codec='libx264') as writer:
            print(f"Saving video to {output_path}")
            for frame in frames:
                writer.append_data(frame)
    


    counts = {}
    for id in objects_detected.keys():
        if id == 'final_frame_for_video':
            continue
        if len(objects_detected[id]['frames']) < min_frames_for_track:
            continue
        counts[id] = {}
        counts[id]['video_path'] = video_path
        first_x1, first_y1, first_x2, first_y2 = [int(x) for x in objects_detected[id]['boxes'][0]]
        last_x1, last_y1, last_x2, last_y2 = [int(x) for x in objects_detected[id]['boxes'][-1]]

        centers_x = [(box[0] + box[2]) / 2 for box in objects_detected[id]['boxes']]
        window = min(3, len(centers_x))
        start_x = float(np.mean(centers_x[:window]))
        end_x = float(np.mean(centers_x[-window:]))
        delta_x = end_x - start_x
        min_displacement = max(width * min_displacement_frac, min_displacement_px)

        if abs(delta_x) >= min_displacement:
            direction = 'Right' if delta_x > 0 else 'Left'
        else:
            direction = 'Unknown'
        counts[id]['direction'] = direction

        # Determine if the fish entered or exited the screen.
        if objects_detected[id]['frames'][0] <= 1:
            entered_frame = 'False'
        else:
            entered_frame = 'True'
        if objects_detected[id]['frames'][-1] >= objects_detected['final_frame_for_video'] - 1:
            exited_frame = 'False'
        else:
            exited_frame = 'True'
        
        # Detemine side of first and last detection. 
        if (first_x1 + first_x2) / 2 < width * 0.5:
            first_side = 'Left'
        elif (first_x1 + first_x2) / 2 > width * 0.5:
            first_side = 'Right'
        else:
            first_side = 'None'
        if (last_x1 + last_x2) / 2 < width * 0.5:
            last_side = 'Left'
        elif (last_x1 + last_x2) / 2 > width * 0.5:
            last_side = 'Right'
        else:
            last_side = 'None'


        counts[id]['first_side'] = first_side
        counts[id]['last_side'] = last_side
        counts[id]['entered_frame'] = entered_frame
        counts[id]['exited_frame'] = exited_frame
        
        # Per-class scores: average class probabilities over all frames (includes Background).
        class_probs_arr = np.array(objects_detected[id]['class_probs'])
        avg_probs = class_probs_arr.mean(axis=0)
        counts[id]['class_scores'] = {
            id2label[c]: float(avg_probs[c]) for c in range(len(id2label))
        }
        counts[id]['top_class'] = id2label[int(np.argmax(avg_probs))]

    return counts
        