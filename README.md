# Focus Detection System with Smart Lighting

A real-time face recognition neural network using PyTorch, NumPy, computer vision, and Geovee api for the light changing to determine if you're "Locked In" or "Geeked" , automatically adjusting Govee smart lights based on your state.

<div align="center">

### 📸 Example Training Images

| Geeked Example |  Locked In Example |
|:---:|:---:|
| <img width="352" alt="Geeked Example" src="https://github.com/user-attachments/assets/0355949a-c966-44c2-aef1-30a99a3be9d2" /> | <img width="352" alt="Locked In Example" src="https://github.com/user-attachments/assets/8c4a8755-0985-400d-9228-da4fca6d2f0a" /> |

</div>


<div align="center">

## Example usage

https://github.com/user-attachments/assets/211f5d47-ec80-4a24-a07c-e3e0c4fcfc89

</div>


## Overview

This project combines OpenCV face detection with a ResNet-18 neural network to monitor your focus state through your webcam. When you're detected as "Geeked" (distracted), the lights turn red. When you're "Locked In" (focused), the lights turn white.

## Features

- Real-time face detection using Haar Cascade classifiers
- Binary image classification (Locked In vs Geeked) using PyTorch and ResNet-18
- Automatic Govee smart light control based on detected state
- Optimized performance (processes every 25th frame to reduce CPU load)
- Visual feedback with bounding boxes and labels

## Prerequisites

- Python 3.7+
- Webcam
- Govee smart lights with API access
- Govee API Key and Device IDs

## Installation

1. Clone this repository
2. Install required dependencies:

```bash
pip install opencv-python torch torchvision pillow pandas python-dotenv requests numpy
```

3. Download the Haar Cascade file:
   - Download `haarcascade_frontalface_default.xml` from the [OpenCV GitHub repository](https://github.com/opencv/opencv/tree/master/data/haarcascades)
   - Place it in the project root directory

4. Create a `.env` file in the project root with your Govee credentials:

```env
GOVEE_API_KEY=your_api_key_here
DEVICE1_ID=your_device1_id
DEVICE2_ID=your_device2_id
DEVICE3_ID=your_device3_id
MODEL=your_govee_model
```

## Project Structure

```
├── Main.py                               # Main webcam loop and face detection
├── Model.py                              # Neural network training script
├── Prediction.py                         # Inference and light control
├── LightChange.py                        # Govee API integration
├── DataSet.csv                           # Training data labels
├── Images/                               # Training images directory
├── haarcascade_frontalface_default.xml   # Face detection model
├── best_model.pth                        # Trained model weights
└── .env                                  # Environment variables
```

## Usage

### Training the Model

1. Prepare your dataset:
   - Create an `Images/` directory
   - Add labeled images of yourself in "Locked In" and "Geeked" states
   - Create `DataSet.csv` with format: `filename,label` (0 = Locked In, 1 = Geeked)

2. Run the training script:

```bash
python Model.py
```

The script will:
- Load and augment your training data
- Train a ResNet-18 model for 5 epochs
- Save the best model as `best_model.pth`

### Running the Detection System

```bash
python Main.py
```

The system will:
- Activate your webcam
- Detect your face in real-time
- Classify your state every 25 frames
- Automatically adjust your Govee lights
- Display visual feedback on screen

Press `ESC` to exit.

## How It Works

### Face Detection
Uses OpenCV's Haar Cascade classifier to detect faces in the webcam feed with an expanded bounding box for better context.

### Classification
A fine-tuned ResNet-18 model processes the detected face region and classifies it as either:
- **Locked In** (0): Focused state → White lights
- **Geeked** (1): Distracted state → Red lights

### Light Control
The Govee API is called to change light colors based on the prediction:
- Red (RGB: 150, 0, 0) for "Geeked"
- White (RGB: 244, 244, 244) for "Locked In"

## Model Details

- **Architecture**: ResNet-18 (pretrained on ImageNet)
- **Classes**: 2 (Binary classification)
- **Input Size**: 224×224 RGB images
- **Training**: 80/20 train/validation split
- **Optimizer**: Adam (lr=1e-4)
- **Loss Function**: Cross-Entropy Loss
- **Data Augmentation**: Random flips, color jitter, rotation

## Configuration

### Adjusting Detection Sensitivity

In `Main.py`, modify the frame processing interval:
```python
if count % 25 == 0:  # Change 25 to process more/fewer frames
```

### Changing Light Colors

In `LightChange.py`, modify the RGB values:
```python
"value": {"r": 150, "g": 0, "b": 0}  # Red
"value": {"r": 244, "g": 244, "b": 244}  # White
```

### Using Multiple Devices

Uncomment additional device calls in `Prediction.py`:
```python
LightChange.changeRed(DEVICE1_ID)
LightChange.changeRed(DEVICE2_ID)
```

## Credits

- **Face Detection**: [Haar Cascade classifiers](https://github.com/opencv/opencv) from OpenCV
- **Deep Learning Framework**: PyTorch
- **Pretrained Model**: ResNet-18 from torchvision

## Troubleshooting

**Cascade file not loading**: Ensure `haarcascade_frontalface_default.xml` is in the project root

**Camera not found**: Check that your webcam is connected and not in use by another application

**Model not found**: Run `Model.py` first to train and generate `best_model.pth`

**Lights not responding**: Verify your `.env` credentials and ensure devices are online

## License

This project is open source and available under the MIT License.

## Future Improvements

- Add more granular focus states
- Implement focus tracking and analytics
- Create a web dashboard for statistics
- Add audio alerts for state changes
- Support for additional smart home devices
