# Motorcycle Detection using NVIDIA Jetson Nano


This project is a motorcycle detection system built using NVIDIA Jetson Nano and the [jetson-inference](https://github.com/dusty-nv/jetson-inference) repository. It leverages a Single Shot MultiBox Detector (SSD) model trained on a subset of the Open Images Dataset to detect motorcycles in images, drawing bounding boxes around detected objects.


---


## Features


- Detects motorcycles in still images
- Trained using SSD with MobileNet backbone
- Runs inference on Jetson Nano with GPU acceleration
- Outputs image with bounding box drawn on detected motorcycle(s)


---


## Requirements


- NVIDIA Jetson Nano (set up with JetPack)
- Docker (installed and configured)
- `jetson-inference` repository cloned:
  ```bash
  git clone --recursive https://github.com/dusty-nv/jetson-inference
  cd jetson-inference


## Dataset
This project uses a custom dataset based on the Open Images Dataset. The dataset should be formatted in Pascal VOC format under the following structure:


data/
└── motorcycle/
    ├── Annotations/
    ├── ImageSets/
    │   └── Main/
    └── JPEGImages/


## Place your dataset under: jetson-inference/python/training/detection/ssd/data/motorcycle/.


## Training the Model
Run the Docker container:


cd ~/jetson-inference/
./docker/run.sh


## Navigate to the SSD training folder:


cd python/training/detection/ssd
## Train the SSD model:


python3 train_ssd.py --dataset-type=voc \
                     --data=data/motorcycle \
                     --model-dir=models/motorcycle
## Convert the trained PyTorch model to ONNX:




python3 onnx_export.py --model-dir=models/motorcycle
Running Inference


Set the model path:


export NET=~/jetson-inference/python/training/detection/ssd/models/motorcycle
Run detection:


detectnet \
  --model=$NET/ssd-mobilenet.onnx \
  --labels=$NET/labels.txt \
  --input-blob=input_0 \
  --output-cvg=scores \
  --output-bbox=boxes \
  data/motorcycle/test/biketest.png output3.jpg
This command will analyze the input image and save an output image (output3.jpg) with a bounding box drawn around the detected motorcycle(s).
