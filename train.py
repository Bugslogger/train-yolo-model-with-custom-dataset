from ultralytics import YOLO

# load a yolo modal
model = YOLO("yolo11s.pt")
# model.tune(use_ray=True)

# train the model
train_results = model.train(
    data='dataset.yaml', # path to dataset YAML
    epochs=80, # number of training epochs
    imgsz=640, # training image size
    device="cpu", # device used for training model
    # space={"lr0": tune.uniform(1e-5, 1e-1)},
)

# Evaluate model performance on the validation set
metrics = model.val()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model