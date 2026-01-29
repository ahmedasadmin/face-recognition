# ArcFace-based Face Recognition System

Real-time face recognition application using **ArcFace** embeddings + **SVM** classifier.

This project implements a complete pipeline for face recognition:
- face embedding extraction using **InsightFace ArcFace** model
- SVM-based classification with probability & cosine similarity thresholding
- real-time video / image folder processing with bounding box visualization

<p align="center">
  <img src="https://github.com/user/repo/raw/main/docs/demo.gif" alt="Demo" width="70%"/>
  <!-- replace with actual screenshot / gif later -->
</p>

## ✨ Features

- High-accuracy face embeddings via **ArcFace** (InsightFace)
- SVM classifier with probability & prototype similarity filtering
- Unknown face rejection using dual thresholds (`prob_threshold` + `sim_threshold`)
- Modular design: embedder, data loader, trainer, recognizer, renderer
- Supports video files and image folders as input
- Clean visualization with OpenCV

## Project Structure

```text
.
├── src/
│   ├── utils.py                 # FrameSource, VideoSource, ImageFolderSource, Renderer
│   ├── archface_embeddings.py   # ArchFaceEmbedder wrapper
│   └── ... 
├── models/
│   ├── svm_arcface.pkl          # trained SVM model
│   ├── label_encoder.pkl
│   └── arcface_prototypes.pkl   # class prototypes (mean embeddings)
├── data/                        # (not in git – your dataset)
│   └── your_dataset/
│       ├── person1/
│       ├── person2/
│       └── ...
├── video.mp4                    # example test video
├── main.py                      # entry point – inference
├── train.py                     # (you should add) training script
├── requirements.txt
└── README.md

# Recommended installation (use virtualenv / conda)
pip install -r requirements.txt

git clone https://github.com/YOUR_USERNAME/face-recognition-arcface.git
cd face-recognition-arcface
