from pylabel import importer
from ultralytics import YOLO

ds = importer.ImportCoco(path="train/out.json", path_to_images="JPEGImages")
ds.splitter.GroupShuffleSplit(train_pct=.7, val_pct=.1, test_pct=.2)
ds.export.ExportToYoloV5(copy_images=True, use_splits=True)


model = YOLO("yolov8n.pt")

model.train(data="training/dataset.yaml", epochs=100, imgsz=640, optimizer="AdamW", batch=32, patience=10)
