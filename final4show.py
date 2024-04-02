import cv2
import torch
import time
import numpy as np
from ultralytics import YOLO
import math

# Load the YOLOv8s model
model = YOLO("yolov8s.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]



#model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

#----------------------------------------------------------------------



def finding_gradient(dot1, dot2):
  try:
     a=(dot2[1]-dot1[1])/(dot2[0]-dot1[0])
  except ZeroDivisionError:
     a=(dot2[1]-dot1[1])/0.0001
  return a

def evaluate_angle(dots_list):
  if len(dots_list) < 3:
    return
  dot1, dot2, dot3 = dots_list[-3:]
  print("dots list ",dots_list)
  m1 = finding_gradient(dot1, dot2)
  m2 = finding_gradient(dot1, dot3)
  angleR = math.atan((m2-m1)/(1+(m2*m1)))
  angleD = round(math.degrees(angleR))
  return angleD


def analyze_row_and_visualize(image, row_percent, igm):
  """
  This function analyzes and visualizes a specific row in an image, 
  providing guidance based on the largest gap available.

  Args:
      image: The input image as a NumPy array.
      row_percent: The row index percentage (0-100).
      igm: (Purpose unclear in this context).

  Returns:
      The modified image with visualization and direction text.
  """

  row_index = math.ceil(image.shape[0] - (row_percent/100) * image.shape[0])
  # Get the specific row
  row = image[row_index, :]

  # Analyze gaps and obstacles
  current_pixel = row[0]  # Start with the first pixel
  gap_count = 0
  obstacle_count = 0
  consecutive_black_starts = []  # Track start of consecutive black pixels
  consecutive_black_lengths = []  # Track length of consecutive black pixels

  for i, pixel in enumerate(row[1:]):
    if pixel == current_pixel:
      # Consecutive pixels of the same color
      if current_pixel == 0:
        gap_count += 1
      else:
        obstacle_count += 1
    else:
      # Encountered a different color
      if current_pixel == 0:
        # Record previous sequence (consecutive black pixels)
        if gap_count > 0:
          consecutive_black_starts.append(i - gap_count)
          consecutive_black_lengths.append(gap_count)
      gap_count = 0 if pixel == 255 else 1  # Reset count for new color (black or white)
      obstacle_count = 0 if pixel == 0 else 1  # Reset count for new color (black or white)
      current_pixel = pixel

  # Handle the last sequence
  if current_pixel == 0:
    if gap_count > 0:
      consecutive_black_starts.append(len(row) - 1 - gap_count)
      consecutive_black_lengths.append(gap_count)

  # Find the largest gap
  largest_gap_index = None
  largest_gap_length = 0
  for i, length in enumerate(consecutive_black_lengths):
    if length > largest_gap_length:
      largest_gap_index = i
      largest_gap_length = length

  # Only visualize and calculate path for the largest gap (if any)
  if largest_gap_index is not None:
    start = consecutive_black_starts[largest_gap_index]
    length = consecutive_black_lengths[largest_gap_index]

    # Calculate midpoint based on actual black pixel length
    midpoint = start + int(length / 2)

    # Draw arrow from last row midpoint to largest gap midpoint
    last_row_midpoint = int(image.shape[1] / 2)
    cv2.arrowedLine(image, (last_row_midpoint, image.shape[0] - 1),
                    (start + int(length / 2), row_index + 1), (255), 2)

    # Define the points for angle calculation (assuming function exists)
    point1 = (last_row_midpoint, image.shape[0] - 1)
    point2 = (320, row_index)
    point3 = (start + int(length / 2), row_index)

    # Call function to calculate angle (replace with your implementation)
    angle_deg = evaluate_angle([point1, point2, point3])

    aa = (angle_deg)
    if aa < 0:
      print("go left by ", abs(aa))
    elif aa == 0:
      print("go straight")
    else:
      print("go right by ", abs(aa))

    # Display the angle value next to the black sequence midpoint
    cv2.putText(image, str(aa), [start + int(length / 2), 20], cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255), 1)
  return image





#----------------------------------------------------------------------------

# Open up the video capture from a webcam
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('C:/Users/el fares/Downloads/TML/video.mp4')
#cap = cv2.VideoCapture('video3.mp4')



while cap.isOpened():

    success, img = cap.read()
    frame=img

    start = time.time()
    

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    
    # Apply input transforms
    input_batch = transform(img).to(device)

    # Prediction and resize to original resolution
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    

    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)


    #odd ---------------------------------------------------------------------
    results = model(frame, stream=True)
    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
            #print("x1",x1,"y1",y1,"x2",x2,"y2",y2)
            #centroid_x =int((x1 + x2) / 2) 
            #centroid_y =int( (y1 + y2) / 2)
            #depth_value = 1/depth_map[centroid_y, centroid_x]
            

            

            # put box in cam
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)
            #cv2.putText(frame, str(depth_value), [centroid_x,centroid_y], font, 0.5, color, thickness)
            #cv2.putText(depth_map, classNames[cls], org, font, fontScale, color, thickness)
            


    end = time.time()
    totalTime = end - start
    fps = 1 / totalTime

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    depth_map = (depth_map*255).astype(np.uint8)
    
    obstacle_threshold = 200 
    obstacle_mask = cv2.threshold(depth_map, obstacle_threshold, 255, cv2.THRESH_BINARY)[1]
    #depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)



    dim = (640, 480)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    depth_map = cv2.resize(depth_map, dim, interpolation=cv2.INTER_AREA)

    end = time.time()
    totalTime = end - start
    fps = 1 / totalTime

    #cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    #cv2.imshow('Image', frame)
    cv2.imshow('Depth Map', depth_map)
    #cv2.imshow('BINARY',obstacle_mask)
    cv2.imshow('gaps',analyze_row_and_visualize(obstacle_mask,20,frame))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
