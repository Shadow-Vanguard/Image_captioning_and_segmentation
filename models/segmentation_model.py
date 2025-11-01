
import os
import cv2
import numpy as np
from ultralytics import YOLO

def random_color(seed: int):
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(0, 255, size=3)]

class Segmenter:
    def __init__(self, model_name="yolov8m-seg.pt", score_thresh: float = 0.5):
        """
        model_name: YOLOv8 segmentation model file (n/s/m/l/x)
        Examples: yolov8n-seg.pt (tiny), yolov8s-seg.pt (small), 
                  yolov8m-seg.pt (medium), yolov8l-seg.pt (large), yolov8x-seg.pt (extra large)
        score_thresh: minimum confidence threshold for detections
        """
        self.model = YOLO(model_name)
        self.score_thresh = score_thresh

    def segment(self, image_path: str, out_dir: str):
        results = self.model.predict(image_path, conf=self.score_thresh)

        # YOLOv8 gives list of results
        result = results[0]

        # original image
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        H, W = bgr.shape[:2]
        overlay = bgr.copy()
        instances = []

        # iterate over detections
        for i, (box, cls, conf) in enumerate(
            zip(result.boxes.xyxy.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
                result.boxes.conf.cpu().numpy())
        ):
            class_name = result.names[int(cls)]
            color = random_color(i)

            # handle mask if segmentation available
            if result.masks is not None:
                mask = result.masks.data[i].cpu().numpy()  # [h, w] float
                mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                mask_bin = (mask_resized > 0.5).astype(np.uint8)

                colored_mask = np.zeros_like(bgr, dtype=np.uint8)
                colored_mask[mask_bin == 1] = color
                overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)

            # draw bbox
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # draw label
            label_txt = f"{class_name} {conf:.2f}"
            cv2.putText(overlay, label_txt, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            instances.append({"class": class_name, "score": float(conf)})

        # save segmented image
        base = os.path.basename(image_path)
        name, ext = os.path.splitext(base)
        out_name = f"{name}_segmented{ext}"
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, overlay)

        return out_name, instances
