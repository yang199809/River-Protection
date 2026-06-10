# -*- coding: utf-8 -*-
"""
YOLO11m-seg inference script without class names and confidence scores.

功能：
1. 调用 Ultralytics YOLO11m-seg 或你训练好的 YOLO11m-seg 权重；
2. 对测试图像文件夹进行批量推理；
3. 将推理可视化结果保存到指定输出文件夹；
4. 输出图像中不显示类别名称和置信度。

示例：
python
                  yolo11m_seg_infer_no_label_conf.py
                   \
    --weights runs/segment/train/weights/
                  best.pt
                   \
    --source dataset/images/test \
    --out_dir dataset/predict_results \
    --imgsz 640 \
    --conf 0.25 \
    --device 0
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO11m-seg inference without class names and confidence scores."
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained YOLO11m-seg weights, e.g., runs/segment/train/weights/best.pt"
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to test image folder."
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to save visualized prediction results."
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size. Default: 640"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold. Default: 0.25"
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold. Default: 0.7"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device, e.g., '0' for GPU 0, 'cpu' for CPU. Default: 0"
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search images in subfolders."
    )

    parser.add_argument(
        "--line_width",
        type=int,
        default=2,
        help="Bounding box line width. Default: 2"
    )

    parser.add_argument(
        "--hide_boxes",
        action="store_true",
        help="Only draw segmentation masks and hide bounding boxes."
    )

    return parser.parse_args()


def collect_images(source_dir: Path, recursive: bool = False):
    if recursive:
        image_paths = [
            p for p in source_dir.rglob("*")
            if p.suffix.lower() in IMG_SUFFIXES
        ]
    else:
        image_paths = [
            p for p in source_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_SUFFIXES
        ]

    return sorted(image_paths)


def main():
    args = parse_args()

    source_dir = Path(args.source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder does not exist: {source_dir}")

    image_paths = collect_images(source_dir, recursive=args.recursive)

    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in: {source_dir}")

    model = YOLO(args.weights)

    print(f"Loaded model: {args.weights}")
    print(f"Found {len(image_paths)} images.")
    print(f"Saving results to: {out_dir}")

    for img_path in image_paths:
        results = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False
        )

        result = results[0]

        # 核心设置：
        # labels=False：不显示类别名称
        # conf=False：不显示置信度
        # masks=True：显示分割掩膜
        # boxes=True/False：是否显示检测框
        vis_img = result.plot(
            labels=False,
            conf=False,
            masks=True,
            boxes=not args.hide_boxes,
            line_width=args.line_width
        )

        # result.plot() 输出为 RGB 格式，cv2.imwrite 需要 BGR 格式
        vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

        save_path = out_dir / img_path.name
        cv2.imwrite(str(save_path), vis_img_bgr)


print("Inference finished.")

if __name__ == "__main__":
    main()