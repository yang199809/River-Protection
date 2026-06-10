# -*- coding: utf-8 -*-
"""
align_yolo_predictions_to_coco.py

功能：
    将 YOLO/Ultralytics 导出的 COCO-style predictions.json 中的字符串型 image_id
    对齐为 test2017.json 中 images[].id 的整数型 COCO image_id，
    以便用于 pycocotools 的 COCO bbox/segm 评价。

适用场景：
    - test2017.json:
        images: [{"id": 0, "file_name": "xxx.jpg", ...}, ...]
    - predictions.json:
        [{"image_id": "xxx", "file_name": "xxx.jpg", "bbox": ..., "score": ..., "segmentation": ...}, ...]

核心原则：
    不修改 test2017.json，只修改 predictions.json。
    因为 test2017.json 中 annotations[].image_id 已经引用 images[].id，
    修改标注文件风险更高；预测文件只需把 image_id 映射为对应整数 id。
"""

import argparse
import json
import os
from copy import deepcopy
from typing import Any, Dict, List, Tuple


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))


def stem(path_or_name: str) -> str:
    """返回文件名主干：/a/b/xxx.jpg -> xxx"""
    return os.path.splitext(os.path.basename(str(path_or_name)))[0]


def basename(path_or_name: str) -> str:
    """返回文件名：/a/b/xxx.jpg -> xxx.jpg"""
    return os.path.basename(str(path_or_name))


def build_image_mapping(coco_gt: Dict[str, Any]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    从 COCO GT 的 images 字段构建两个映射：
        file_name -> id
        file_stem -> id
    """
    if "images" not in coco_gt:
        raise KeyError("GT JSON 中没有 'images' 字段，无法建立 file_name 到 image_id 的映射。")

    file_to_id: Dict[str, int] = {}
    stem_to_id: Dict[str, int] = {}

    duplicate_files = []
    duplicate_stems = []

    for img in coco_gt["images"]:
        if "id" not in img or "file_name" not in img:
            raise KeyError("GT JSON 的 images 元素必须包含 'id' 和 'file_name' 字段。")

        img_id = img["id"]
        file_name = basename(img["file_name"])
        file_stem = stem(img["file_name"])

        if file_name in file_to_id:
            duplicate_files.append(file_name)
        if file_stem in stem_to_id:
            duplicate_stems.append(file_stem)

        file_to_id[file_name] = img_id
        stem_to_id[file_stem] = img_id

    if duplicate_files:
        raise ValueError(f"GT 中存在重复 file_name，无法唯一映射。示例：{duplicate_files[:5]}")
    if duplicate_stems:
        raise ValueError(f"GT 中存在重复文件主干名，无法唯一映射。示例：{duplicate_stems[:5]}")

    return file_to_id, stem_to_id


def resolve_prediction_image_id(
    det: Dict[str, Any],
    gt_file_to_id: Dict[str, int],
    gt_stem_to_id: Dict[str, int],
) -> Tuple[int, str]:
    """
    为单条预测结果找到 COCO GT 中对应的整数 image_id。

    优先级：
        1. det["file_name"] 精确匹配 GT file_name
        2. det["file_name"] 的 basename 匹配 GT file_name
        3. det["image_id"] 若为整数且已在 GT id 集合中，直接保留
        4. det["image_id"] 若为字符串主干名，匹配 GT file_stem
        5. det["image_id"] 若为带扩展名文件名，匹配 GT file_name
    """
    gt_ids = set(gt_file_to_id.values())

    # 1/2. 优先使用 file_name，因为它是连接预测和 GT 最清晰的字段
    if "file_name" in det and det["file_name"] is not None:
        fn = basename(det["file_name"])
        if fn in gt_file_to_id:
            return gt_file_to_id[fn], "file_name"

        fs = stem(det["file_name"])
        if fs in gt_stem_to_id:
            return gt_stem_to_id[fs], "file_name_stem"

    # 3/4/5. 回退使用 image_id
    if "image_id" not in det:
        raise KeyError("预测结果中缺少 'image_id' 字段，且无法通过 file_name 对齐。")

    raw_id = det["image_id"]

    if isinstance(raw_id, int) and raw_id in gt_ids:
        return raw_id, "already_int"

    raw_str = str(raw_id)
    raw_base = basename(raw_str)
    raw_stem = stem(raw_str)

    if raw_base in gt_file_to_id:
        return gt_file_to_id[raw_base], "image_id_as_file_name"

    if raw_stem in gt_stem_to_id:
        return gt_stem_to_id[raw_stem], "image_id_as_stem"

    raise ValueError(
        "无法为预测结果匹配 GT 图像："
        f"image_id={det.get('image_id')}, file_name={det.get('file_name')}"
    )


def align_predictions(
    gt_json_path: str,
    pred_json_path: str,
    out_json_path: str,
    keep_file_name: bool = False,
    strict: bool = True,
) -> None:
    """
    对齐预测文件中的 image_id，并保存新的 predictions_aligned.json。

    参数：
        keep_file_name:
            True  - 输出中保留 file_name 字段，便于排查；
            False - 输出为更标准的 COCO result 格式，去掉 file_name。
        strict:
            True  - 发现无法匹配的预测直接报错；
            False - 跳过无法匹配的预测，并输出 unmatched 统计。
    """
    coco_gt = load_json(gt_json_path)
    preds = load_json(pred_json_path)

    if not isinstance(preds, list):
        raise TypeError("predictions.json 应为 COCO result list，即最外层是 list。")

    gt_file_to_id, gt_stem_to_id = build_image_mapping(coco_gt)

    aligned: List[Dict[str, Any]] = []
    unmatched: List[Dict[str, Any]] = []
    method_counter: Dict[str, int] = {}

    # COCO result 标准字段。segm 评价需要 segmentation；bbox 评价需要 bbox。
    standard_keys = {"image_id", "category_id", "bbox", "score", "segmentation"}

    for det in preds:
        try:
            new_image_id, method = resolve_prediction_image_id(det, gt_file_to_id, gt_stem_to_id)
        except Exception as e:
            if strict:
                raise
            bad = deepcopy(det)
            bad["_align_error"] = str(e)
            unmatched.append(bad)
            continue

        new_det = deepcopy(det)
        new_det["image_id"] = int(new_image_id)

        # 建议输出标准 COCO result 字段，避免 file_name 等额外字段干扰某些自定义评估脚本
        if not keep_file_name:
            new_det = {k: v for k, v in new_det.items() if k in standard_keys}

        aligned.append(new_det)
        method_counter[method] = method_counter.get(method, 0) + 1

    save_json(aligned, out_json_path)

    if unmatched:
        unmatched_path = os.path.splitext(out_json_path)[0] + "_unmatched.json"
        save_json(unmatched, unmatched_path)
    else:
        unmatched_path = None

    gt_image_ids = {img["id"] for img in coco_gt["images"]}
    pred_image_ids = {det["image_id"] for det in aligned}

    print("=" * 80)
    print("COCO prediction image_id alignment finished.")
    print(f"GT images:              {len(coco_gt['images'])}")
    print(f"GT annotations:         {len(coco_gt.get('annotations', []))}")
    print(f"Original predictions:   {len(preds)}")
    print(f"Aligned predictions:    {len(aligned)}")
    print(f"Unmatched predictions:  {len(unmatched)}")
    print(f"Matched image count:    {len(pred_image_ids & gt_image_ids)} / {len(gt_image_ids)}")
    print(f"Mapping methods:        {method_counter}")
    print(f"Saved to:               {out_json_path}")
    if unmatched_path:
        print(f"Unmatched saved to:     {unmatched_path}")
    print("=" * 80)

    # 强校验：输出中的 image_id 必须全部出现在 GT images[].id 中
    illegal_ids = sorted(pred_image_ids - gt_image_ids)
    if illegal_ids:
        raise ValueError(f"输出预测中仍存在 GT 不包含的 image_id。示例：{illegal_ids[:10]}")


def evaluate_with_pycocotools(gt_json_path: str, pred_json_path: str, eval_type: str = "segm") -> None:
    """
    可选：直接调用 pycocotools 评价。
    eval_type:
        - 'bbox'：检测框评价
        - 'segm'：实例分割掩膜评价
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(pred_json_path)

    coco_eval = COCOeval(coco_gt, coco_dt, eval_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align YOLO/Ultralytics predictions image_id with COCO GT images[].id."
    )
    parser.add_argument("--gt", required=True, help="COCO GT annotation JSON, e.g., test2017.json")
    parser.add_argument("--pred", required=True, help="YOLO prediction JSON, e.g., predictions.json")
    parser.add_argument("--out", required=True, help="Output aligned prediction JSON")
    parser.add_argument(
        "--keep-file-name",
        action="store_true",
        help="Keep file_name in output predictions for debugging. Default: remove it.",
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Skip unmatched predictions instead of raising errors.",
    )
    parser.add_argument(
        "--eval",
        choices=["bbox", "segm"],
        default=None,
        help="Optionally run pycocotools evaluation after alignment.",
    )

    args = parser.parse_args()

    align_predictions(
        gt_json_path=args.gt,
        pred_json_path=args.pred,
        out_json_path=args.out,
        keep_file_name=args.keep_file_name,
        strict=not args.non_strict,
    )

    if args.eval is not None:
        evaluate_with_pycocotools(args.gt, args.out, args.eval)


if __name__ == "__main__":
    main()
