from ultralytics.data.converter import convert_coco
from ultralytics import YOLO
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer
from ultralytics import RTDETR



'''
convert_coco(labels_dir='/home/yangshuang/ultralytics/Broken/annotations',
             save_dir='/home/yangshuang/ultralytics/Broken/output_dir',
             use_segments=True, use_keypoints=False, cls91to80=False)
'''







# train

# YOLOv11m
'''
model = YOLO("yolo11m-seg.yaml")
# model = YOLO("yolo11n.pt")
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")
results = model.train(data="ssdd-seg.yaml", epochs=300, batch=4, imgsz=512, pretrained=False, seed=42,
                      optimizer='AdamW', lr0=0.0005)
'''
'''
model = YOLO("yolo11m-seg.yaml")
# model = YOLO("yolo11n.pt")
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")
results = model.train(data="hrsid-seg.yaml", epochs=300, batch=4, imgsz=800, pretrained=False, seed=42,
                      optimizer='AdamW', lr0=0.0005)
'''


# YOLOv12m

'''
model = YOLO("yolo12m-seg.yaml")
# model = YOLO("yolo11n.pt")
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")
results = model.train(data="ssdd-seg.yaml", epochs=300, batch=4, imgsz=512, pretrained=False, seed=42,
                      optimizer='AdamW', lr0=0.0005)
'''
'''
model = YOLO("yolo12m-seg.yaml")
# model = YOLO("yolo11n.pt")
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")
results = model.train(data="hrsid-seg.yaml", epochs=300, batch=4, imgsz=800, pretrained=False, seed=42,
                      optimizer='AdamW', lr0=0.0005)
'''


# YOLOv9c
'''
model = YOLO("yolov9c-seg.yaml")
# model = YOLO("yolo11n.pt")
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")
results = model.train(data="ssdd-seg.yaml", epochs=300, batch=4, imgsz=512, pretrained=False, seed=42,
                      optimizer='AdamW', lr0=0.0005)
'''

'''
model = YOLO("yolov9c-seg.yaml")
# model = YOLO("yolo11n.pt")
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")
results = model.train(data="hrsid-seg.yaml", epochs=300, batch=4, imgsz=800, pretrained=False, seed=42,
                      optimizer='AdamW', lr0=0.0005)
'''

# YOLOv8m
'''
model = YOLO("yolov8m-seg.yaml")
# model = YOLO("yolo11n.pt")
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")
results = model.train(data="ssdd-seg.yaml", epochs=300, batch=4, imgsz=512, pretrained=False, seed=42,
                      optimizer='AdamW', lr0=0.0005)
'''

'''
model = YOLO("yolov8m-seg.yaml")
# model = YOLO("yolo11n.pt")
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")
results = model.train(data="hrsid-seg.yaml", epochs=300, batch=4, imgsz=800, pretrained=False, seed=42,
                      optimizer='AdamW', lr0=0.0002)
'''

# YOLOEm
'''
model = YOLO("yoloe-v8m-seg.yaml")
model = YOLOE("yoloe-v8m-seg-pf.pt")
results = model.train(data="ssdd-seg.yaml", trainer=YOLOEPESegTrainer, epochs=80, batch=4, imgsz=512, pretrained=True, seed=42,
                      optimizer='AdamW', lr0=0.0005, patience=10)
'''
'''
model = YOLO("yoloe-v8m-seg.yaml")
model = YOLOE("yoloe-v8m-seg-pf.pt")
results = model.train(data="hrsid-seg.yaml", trainer=YOLOEPESegTrainer, epochs=80, batch=4, imgsz=800, pretrained=True, seed=42,
                      optimizer='AdamW', lr0=0.0005, patience=10)
'''


# YOLOEm
'''
model = YOLO("yoloe-11m-seg.yaml")
model = YOLOE("yoloe-11m-seg-pf.pt")
results = model.train(data="ssdd-seg.yaml", trainer=YOLOEPESegTrainer, epochs=80, batch=4, imgsz=512, pretrained=True, seed=42,
                      optimizer='AdamW', lr0=0.0002, patience=10)
'''

# RT-DETR
'''
model = RTDETR("rtdetr-l.yaml")
model.info()
# model = YOLO("yolo11n.pt")
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")
results = model.train(data="ssdd-seg.yaml", epochs=300, batch=4, imgsz=512, pretrained=False, seed=42,
                      optimizer='AdamW', lr0=0.0005)
'''

'''
model = RTDETR("rtdetr-x.yaml")
model.info()
# model = YOLO("yolo11n.pt")
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")
results = model.train(data="ssdd-seg.yaml", epochs=300, batch=4, imgsz=512, pretrained=False, seed=42,
                      optimizer='AdamW', lr0=0.0005)
'''

'''
model = RTDETR("rtdetr-l.yaml")
model.info()
# model = YOLO("yolo11n.pt")
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")
results = model.train(data="hrsid-seg.yaml", epochs=300, batch=4, imgsz=800, pretrained=False, seed=42,
                      optimizer='AdamW', lr0=0.0005)
'''

'''
model = RTDETR("rtdetr-x.yaml")
model.info()
# model = YOLO("yolo11n.pt")
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")
results = model.train(data="hrsid-seg.yaml", epochs=300, batch=4, imgsz=800, pretrained=False, seed=42,
                      optimizer='AdamW', lr0=0.0005)
'''

'''
model = RTDETR("rtdetr-resnet50.yaml")
model.info()
# model = YOLO("yolo11n.pt")
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")
results = model.train(data="ssdd-seg.yaml", epochs=300, batch=4, imgsz=512, pretrained=False, seed=42,
                      optimizer='AdamW', lr0=0.0005)
'''

'''
# model = YOLO("yolo11m-seg.yaml")
# model = YOLO("yolo11m-seg.pt")
model = YOLO("yolo11m.yaml").load("yolo11m.pt")
results = model.train(data="waterway-seg.yaml", epochs=300, batch=2, imgsz=800, pretrained=True, seed=42,
                      optimizer='AdamW', lr0=0.0005)
'''

'''
model = YOLO("yolo26m-seg.yaml")
# model = YOLO("yolo11n.pt")
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")
results = model.train(data="ssdd-seg.yaml", epochs=300, batch=4, imgsz=512, pretrained=False, seed=42,
                      optimizer='AdamW', lr0=0.0005)
'''

'''
model = YOLO("yolo26m-seg.yaml")
# model = YOLO("yolo11n.pt")
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")
results = model.train(data="hrsid-seg.yaml", epochs=300, batch=4, imgsz=800, pretrained=False, seed=42,
                      optimizer='AdamW', lr0=0.0005)
'''



'''
model = YOLO("yolo11m-seg.yaml")
# model = YOLO("yolo11n.pt")
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")
results = model.train(data="hrsid-seg.yaml", epochs=300, batch=4, imgsz=800, pretrained=False, seed=42,
                      optimizer='AdamW', lr0=0.0005)
'''


'''
# model = YOLO("yolo12m-seg.yaml")
# model = YOLO("yolo11n.pt")
model = YOLO("yolo11m-seg.yaml")
# model = YOLO("yolo11m-seg.pt")
results = model.train(data="Gaofen-seg.yaml", epochs=300, batch=4, imgsz=800, pretrained=False, seed=42,
                      optimizer='AdamW', lr0=0.0005)
'''


# model = YOLO("yolo12m-seg.yaml")
# model = YOLO("yolo11n.pt")
model = YOLO("yolov9c-seg.yaml")
# model = YOLO("yolo11m-seg.pt")
results = model.train(data="/home/yangshuang/ultralytics/ultralytics/cfg/datasets/Gaofen-seg.yaml", epochs=300, batch=4, imgsz=800, pretrained=False, seed=42,
                      optimizer='AdamW', lr0=0.0005)



'''
# model = YOLO("yolo12m-seg.yaml")                                                                       
# model = YOLO("yolo11n.pt")                                                                             
# model = YOLO("yolo11m-seg.yaml")                                                                       
model = YOLO("yolo11s-seg.pt")
results = model.train(data="waterway-seg.yaml", epochs=300, batch=2, imgsz=800, pretrained=True, seed=42,
                      optimizer='AdamW', lr0=0.0002)     
'''



# val
'''
# model = YOLO("/home/shuangy/ultralytics/runs/segment/HRSID/RT-DETR-l_800/weights/best.pt")
# model = RTDETR("/home/shuangy/ultralytics/runs/detect/RT-DETR-ResNet50_512/weights/best.pt")
model = YOLO("/home/shuangy/ultralytics/runs/segment/train-2/weights/best.pt")
# metrics = model.val(data="hrsid-seg.yaml", imgsz=800, batch=1, conf=0.001, iou=0.7, max_det=300, save_json=True)
metrics = model.val(data="ssdd-seg.yaml", imgsz=512, batch=1, conf=0.001, iou=0.7, max_det=300, save_json=True)
metrics.box.map
metrics.box.map50
metrics.box.map
metrics.box.maps
metrics.seg.map
metrics.seg.map50
metrics.seg.map75
metrics.seg.maps

# model = YOLO("yolo12m-seg.yaml")
# model = YOLO("yoloe-11m-seg.yaml")
# model = YOLO("yolo12m-seg.pt")

'''






















