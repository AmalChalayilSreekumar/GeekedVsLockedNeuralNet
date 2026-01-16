# GeekedVsLockedNeuralNet
This program uses PyTorch to create an image recognition neural network that distinguishes between "Geeked" and "Locked" in real time using cv2. When recognizing either "Geeked" or "Locked", given a valid API key and lightbulbs for a GOVEE light bulb will change the colour of the lightbulb. 

## Libraries
To run you will require the following Imports:
pandas
PIL.Image
torch	
torch.utils.data
torchvision.transforms
torch.nn	
requests	
dotenv.load_dotenv
cv2	
matplotlib.pyplot
IPython.display

To install all of them without CUDA installed graphics:
pip install pandas Pillow torch torchvision requests python-dotenv opencv-python matplotlib ipython

## Dependencies

This project uses OpenCV Haar Cascade classifiers.

- `haarcascade_frontalface_default.xml`
  Source: OpenCV GitHub Repository  
  https://github.com/opencv/opencv/tree/master/data/haarcascades
