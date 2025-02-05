# Obstacle Avoidance by Using AI

This repository combines YOLOv8 for object detection and MiDaS for depth estimation to create a system that identifies objects in a video stream and estimates their distance. It also includes a navigation assistance feature based on the depth map.

## Overview

The code utilizes the YOLOv8 object detection model to identify objects in real-time.  Simultaneously, it employs the MiDaS depth estimation model to generate a depth map of the scene.  This depth information is then used to identify obstacles and provide directional guidance (left, right, or straight) for navigation, based on the largest gap detected in a specific row of the depth map.

## Features

*   Real-time Object Detection: Uses YOLOv8 to detect and classify objects in a video stream.
*   Depth Estimation: Employs MiDaS to generate depth maps, providing distance information for each pixel.
*   Obstacle Detection:  Identifies obstacles in the depth map based on a threshold.
*   Navigation Assistance: Analyzes a row in the depth map, finds the largest gap (free space), and provides directional cues (left, right, straight) for navigation.
*   FPS Display: Shows the frames per second (FPS) for performance monitoring.

## Technologies Used

*   Python: The primary programming language.
*   OpenCV (cv2): For image and video processing.
*   PyTorch: Used by both YOLOv8 and MiDaS.
*   Ultralytics YOLOv8: For object detection.
*   MiDaS (Multi-Inference Depth Aggregation System): For depth estimation.
*   NumPy: For numerical operations.

## Installation

1.  Clone the repository:

    git clone https://github.com/hemandsunny/obstacle-avoidance-using-multimodal-deep-learning.git

2.  Install required libraries:

    pip install -r requirements.txt  # If you have a requirements file. Otherwise, install individually:
    pip install opencv-python torch torchvision torchaudio ultralytics numpy
    

3.  Download YOLOv8 model:  The code uses the `yolov8s.pt` model by default. Ensure this file exists in the same directory as the script, or modify the `model = YOLO("yolov8s.pt")` line to point to the correct path.  You can download pretrained YOLOv8 models from the Ultralytics YOLOv8 repository.

4.  Download MiDaS model: The MiDaS models are loaded automatically by the `torch.hub.load` command in the code, so you generally don't need to download them manually.

## Usage

1.  Run the script:
   python main_script.py

2.  Webcam or Video File: The script currently uses the default webcam (index 0). To use a video file, change the `cv2.VideoCapture(0)` line to `cv2.VideoCapture('path/to/your/video.mp4')`.

3.  Navigation: The navigation prompts (Go Left, Go Right, Go Straight) are displayed on the video feed.

4.  Exit: Press 'q' to exit the program.

## Code Explanation

*   Object Detection: The YOLOv8 model detects objects and draws bounding boxes around them, along with class labels and confidence scores.
*   Depth Estimation: The MiDaS model generates a depth map, where each pixel's value represents its distance from the camera.
*   Obstacle Mask: The depth map is thresholded to create a binary mask, where white pixels represent obstacles (close objects) and black pixels represent free space (further away).
*   Navigation Logic: The `analyze_row_and_visualize` function examines a horizontal row in the obstacle mask. It identifies the largest continuous gap (black pixels) and calculates the angle to that gap from the center of the bottom of the frame. This angle is used to provide directional guidance.

## Future Improvements

*   More Robust Navigation: The current navigation logic is based on a single row.  More sophisticated path planning algorithms could be implemented for smoother and more accurate navigation.
*   Distance Measurement:  Calibrate the depth map to get actual distance measurements in meters or other units.
*   User Interface: Add a graphical user interface (GUI) for better control and visualization.
*   Performance Optimization: Explore ways to optimize the code for faster processing, especially for real-time applications.

## Author

Hemand Sunny

## License

NA
