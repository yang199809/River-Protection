import faster_coco_eval
faster_coco_eval.init_as_pycocotools()

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

anno = COCO("/home/shuangy/ultralytics/datasets/ssdd/annotations/test.json") # Load your JSON annotations
# anno = COCO("/home/shuangy/ultralytics/datasets/hrsid/annotations/test2017.json")
# pred = anno.loadRes("/home/shuangy/ultralytics/runs/segment/HRSID/yoloev8m_800/val/predictions.json")   # Load predictions.json
pred = anno.loadRes("/home/shuangy/ultralytics/runs/segment/val/predictions.json")
# val = COCOeval(anno, pred, "bbox")
val = COCOeval(anno, pred, "segm")
val.evaluate()
val.accumulate()
val.summarize()
