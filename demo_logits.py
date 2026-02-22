#!/usr/bin/env python3
"""
Demo: run detection on a video until the first frame with a detection above 0.7,
then print model outputs including logits and softmax class probabilities.
"""
import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

VIDEO_PATH = "/home/quinn/lake_ontario/2.mp4"
CHECKPOINT = "/home/quinn/detr_r50_complete_augmentation/checkpoint-484110"
CONF_THRESH = 0.7
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def main():
    print("Loading model and image processor...")
    model = (
        AutoModelForObjectDetection.from_pretrained(CHECKPOINT, local_files_only=True)
        .to(DEVICE)
        .eval()
    )
    image_processor = AutoImageProcessor.from_pretrained(CHECKPOINT, local_files_only=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {VIDEO_PATH}")

    frame_idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        processed = image_processor(pil_image, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(DEVICE)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, return_dict=True)

        detections = image_processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([[pil_image.height, pil_image.width]]),
            threshold=CONF_THRESH,
        )[0]

        if len(detections["boxes"]) == 0:
            continue

        # First frame with at least one detection above threshold
        print(f"\n--- First frame with detection above {CONF_THRESH} (frame index {frame_idx}) ---\n")

        boxes = detections["boxes"].cpu()
        scores = detections["scores"].cpu()
        labels = detections["labels"].cpu()
        n = len(boxes)
        print(f"Number of detections: {n}")
        print("\nDetections (boxes [x1,y1,x2,y2], score, label):")
        for i in range(n):
            print(f"  {i}: box={boxes[i].tolist()}, score={scores[i].item():.4f}, label={labels[i].item()}")

        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        print(f"\nLogits shape: {logits.shape}")
        print(f"Softmax class probs shape: {probs.shape}")
        print("(batch, num_queries, num_classes) — each row sums to 1 over classes.")

        # Mirror HF post-processor: they only threshold (no NMS). Score per query = max over
        # class probs; often the last class is "no-object" so we use [..., :-1].
        prob_no_null = probs[0, :, :-1]
        scores_per_query = prob_no_null.max(dim=-1).values
        keep = scores_per_query > CONF_THRESH
        query_indices = torch.where(keep)[0]

        logits_kept = logits[0][keep].cpu()
        probs_kept = probs[0][keep].cpu()
        print(f"\nQueries that pass threshold (score > {CONF_THRESH}): indices {query_indices.tolist()}")
        print("Logits / softmax probs for those queries only (one row per detection):")
        for i, q in enumerate(query_indices.tolist()):
            print(f"  detection {i} (query {q}): logits={logits_kept[i].numpy()}, probs={probs_kept[i].numpy()} (sum={probs_kept[i].sum().item():.4f})")

        cap.release()
        return

    cap.release()
    print(f"No frame had a detection with confidence > {CONF_THRESH}")


if __name__ == "__main__":
    main()
