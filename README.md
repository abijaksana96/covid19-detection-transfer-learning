# COVID-19 Detection using Transfer Learning (VGG16)

This project implements a Deep Learning model for detecting COVID-19 from chest X-ray images using Transfer Learning with the VGG16 architecture. The model is deployed as a REST API using FastAPI and Docker.

## Project Overview

The system classifies chest X-ray images into two categories:
- **Normal**
- **COVID-19**

It uses a pre-trained VGG16 model as a feature extractor, followed by custom dense layers for classification. The application provides a web API to upload images and receive predictions with confidence scores.

## Features

- **Transfer Learning**: Utilizes VGG16 pre-trained on ImageNet.
- **FastAPI Backend**: High-performance, easy-to-use API framework.
- **Dockerized**: Fully containerized for easy deployment.
- **Image Preprocessing**: Automatic resizing and normalization of input images.
- **Confidence Scores**: Returns probability distribution for predictions.

## Prerequisites

- [Docker](https://www.docker.com/) installed on your machine.
- [Git](https://git-scm.com/) (optional, for cloning).
- [Git LFS](https://git-lfs.github.com/) (required if cloning the model file).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/abijaksana96/covid19-detection-transfer-learning.git
    cd covid19-detection-transfer-learning
    ```

2.  **Download the Model:**
    The model file `app/model/tf_learning_with_vgg16.h5` is large (>100MB).
    - If you have Git LFS installed, it should be downloaded automatically.
    - Otherwise, ensure you have the model file placed in `app/model/`.

3.  **Build the Docker image:**
    ```bash
    docker build -t covid19-detection .
    ```

## Usage

### Running with Docker

Run the container mapping port 8000:

```bash
docker run -d -p 8000:8000 --name covid-app covid19-detection
```

The API will be available at `http://localhost:8000`.

### Running Locally (without Docker)

1.  Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Run the server:
    ```bash
    uvicorn app.server:app --reload --host 0.0.0.0 --port 8000
    ```

## API Documentation

### 1. Health Check
- **URL**: `/`
- **Method**: `GET`
- **Response**:
    ```json
    {
        "message": "Welcome to the COVID-19 Classification API!"
    }
    ```

### 2. Predict
- **URL**: `/predict`
- **Method**: `POST`
- **Body**: `form-data`
    - `file`: Image file (JPEG, PNG)
- **Response**:
    ```json
    {
        "filename": "xray_image.jpg",
        "predicted_class": "covid",
        "confidence": 0.98,
        "probabilities": {
            "covid": 98.5,
            "normal": 1.5
        }
    }
    ```

## Model Training

The model training process is documented in `notebook/tf-learning.ipynb`.
- **Architecture**: VGG16 (weights frozen) + GlobalAveragePooling2D + Dense Layers.
- **Data Augmentation**: Rotation, shift, shear, zoom, flip.
- **Optimizer**: Adam.
- **Loss Function**: Binary Crossentropy.

## Project Structure

```
.
├── app/
│   ├── model/
│   │   └── tf_learning_with_vgg16.h5  # Trained Model
│   ├── server.py                      # FastAPI Application
│   └── __pycache__/
├── notebook/
│   └── tf-learning.ipynb              # Training Notebook
├── Dockerfile                         # Docker Configuration
├── requirements.txt                   # Python Dependencies
├── .gitignore
└── README.md
```
