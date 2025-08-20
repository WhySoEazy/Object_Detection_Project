# Object Detection Project (Faster R-CNN · PyTorch)

A minimal, end-to-end object detection pipeline using **Faster R-CNN** in **PyTorch**, trained and evaluated on datasets in **Pascal VOC format**. Includes a simple dataset wrapper, training/evaluation scripts, TensorBoard logging, and sample predictions.

---

## Features

- 🚀 **Faster R-CNN** with torchvision backbones  
- 🗂️ **VOC-style dataset** support (`JPEGImages`, `Annotations`, `ImageSets/Main`)  
- 📊 **TensorBoard** logs for losses/metrics  
- 🖼️ **Sample output** (`Prediction.jpg`) to verify inference pipeline  
- 🧪 Separate **train** and **test** scripts for clarity

---

## Repository structure

```
.
├── VOC_Dataset.py            # Pascal VOC dataset handle
├── train_fasterrcnn.py       # Training (and validation) loop
├── test_fasterrcnn.py        # Inference / evaluation script
├── tensorboard/              # (Created at runtime) TensorBoard runs
└── Prediction.jpg            # Example prediction output
```

---

## Requirements

- Python 3.8+
- PyTorch & torchvision (CUDA optional)
- Common libs: `opencv-python`, `numpy`, `pillow`, `tqdm`, `matplotlib`, `tensorboard`

Install (example):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # or cpu wheels
pip install opencv-python numpy pillow tqdm matplotlib tensorboard
```

---

## Dataset setup (Pascal VOC format)

Your dataset directory should follow the standard VOC layout:

```
VOCdevkit/
└── VOC2007/
    ├── Annotations/         # .xml files
    ├── JPEGImages/          # .jpg images
    └── ImageSets/
        └── Main/
            ├── train.txt
            ├── val.txt
            └── test.txt
```

---

## Quick start

### 1. Train

```bash
python train_fasterrcnn.py   --data_path /path/to/VOCdevkit/VOC2007   --epochs 100   --batchs 8   --lr 1e-3   --momentum 0.9
```
You can add checkpoint as `--checkpoint trained_model/last_rcnn.pt` to continue the training loop if you had interupted.

Launch TensorBoard:

```bash
tensorboard --logdir tensorboard
```

### 2. Test / Inference
#### For image

```bash
python test_fasterrcnn.py   --checkpoint trained_model/best_rcnn.pt   --image_path /path/to/images --threshold 0.5
```
The result will be saved as `Prediction.jpg`
#### For video

```bash
python test_fasterrcnn.py   --checkpoint trained_model/best_rcnn.pt   --video_path /path/to/video --threshold 0.5
```
The result will be saved as `resultvideo.mp4`

---

## How it works (high level)

- **Dataset**: `VOC_Dataset.py` parses VOC XMLs → returns `image, target` in torchvision detection format.  
- **Model**: `torchvision.models.detection.fasterrcnn_resnet50_fpn(...)`.  
- **Training**: Computes detection losses (RPN + ROI heads).  
- **Evaluation**: Runs inference, score filtering, visualization, optional mAP.

---

## Results

- Example output: `Prediction.jpg` for image and `resultvideo.mp4` for video.

---

## Troubleshooting

- **CUDA OOM** → lower batch size or image size.  
- **Empty detections** → check class ids, score threshold.  
- **VOC parsing errors** → validate XMLs and `ImageSets` list.

---

## Roadmap

- [ ] Config file for hyperparameters  
- [ ] VOC mAP evaluation  
- [ ] COCO dataset support  
- [ ] Experiment tracking (wandb)

---