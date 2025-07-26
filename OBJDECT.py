from ultralytics import YOLO
import cv2

# Load a pre-trained YOLOv8 model. You can choose sizes ('n', 's', 'm', 'l', 'x').
# 'yolov8n.pt' is the smallest and fastest, but may have lower accuracy.
# 'yolov8x.pt' is the fastest, may have high accuracy.
# model = YOLO('yolov8s.pt')
# model = YOLO('yolov8n.pt')
# model = YOLO('yolov8m.pt')
# model = YOLO('yolov8l.pt')
model = YOLO('yolov8x.pt')

# Option 1: Detect objects in a image
# image_path = 'Trial Image\download6.jpg' 
# try:
#     #Detection on the image
#     results = model.predict(image_path)

#     for result in results:
#         boxes = result.boxes.cpu().numpy()  
#         confidence_scores = boxes.conf
#         class_ids = boxes.cls.astype(int)
#         class_names = result.names

#         #Bunding boxes
#         image = cv2.imread(image_path)
#         for i in range(len(boxes)):
#             x1, y1, x2, y2 = map(int, boxes.xyxy[i])
#             confidence = confidence_scores[i]
#             class_id = class_ids[i]
#             class_name = class_names[class_id]
#             label = f'{class_name}: {confidence:.2f}'

#             color = (255, 0, 0)  # colors BGR
#             cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#         # Display
#         cv2.imshow('Object Detection', image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

# except FileNotFoundError:
#     print(f"Error: Image not found at '{image_path}'")
# except Exception as e:
#     print(f"An error occurred during image detection: {e}")

# Option 2: Detect objects in a video or from a webcam 
video_path = ''  # Actual path to your video, or 0 for webcam
url = 'http://192.168.152.128:8080/video'
try:
    # cap = cv2.VideoCapture(video_path)
    # cap = cv2.VideoCapture(url)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame)

        for result in results:
            boxes = result.boxes.cpu().numpy()
            confidence_scores = boxes.conf
            class_ids = boxes.cls.astype(int)
            class_names = result.names

            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                confidence = confidence_scores[i]
                class_id = class_ids[i]
                class_name = class_names[class_id]
                label = f'{class_name}: {confidence:.2f}'

                color = (0, 224, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('Video Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): #exit
            break

    cap.release()
    cv2.destroyAllWindows()

except FileNotFoundError:
    print(f"Error: Video not found at '{video_path}'")
except Exception as e:
    print(f"An error occurred during video detection: {e}")



# "Droidcam" and "ip camera" to use phone as camera

























# import cv2
# import numpy as np

# # Load YOLO model
# weights_path = "yolov3.weights"  # Path to weights file
# config_path = "yolov3.cfg"       # Path to configuration file
# labels_path = "coco.names"       # Path to COCO dataset labels

# # Load labels
# with open(labels_path, "r") as f:
#     labels = f.read().strip().split("\n")



# # Initialize the network
# net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
# # output_layer_names = [net.getLayerNames()[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# layer_names = net.getLayerNames()
# unconnected_out_layers = net.getUnconnectedOutLayers()

# # Check if unconnected_out_layers is a scalar or list
# if isinstance(unconnected_out_layers, int):
#     output_layer_names = [layer_names[i - 1] for i in unconnected_out_layers]
# else:
#     output_layer_names = [layer_names[i[0] - 1] for i in unconnected_out_layers]


# # Initialize webcam or video
# video_path = "F:\Collage\vivek sis wed 12-12-24\VID-20241213-WA0002.mp4" # Change to a video file path if needed
# # cap = cv2.VideoCapture(video_path)
# cap = cv2.VideoCapture(0)

# # Object detection loop
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     height, width = frame.shape[:2]
#     # Prepare the image for the neural network
#     blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
#     net.setInput(blob)
#     outputs = net.forward(output_layer_names)

#     # Process outputs
#     boxes = []
#     confidences = []
#     class_ids = []

#     for output in outputs:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5:  # Confidence threshold
#                 box = detection[0:4] * np.array([width, height, width, height])
#                 (center_x, center_y, w, h) = box.astype("int")
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#                 boxes.append([x, y, int(w), int(h)])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     # Non-max suppression to remove overlapping boxes
#     indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

#     for i in indices:
#         i = i[0]
#         box = boxes[i]
#         x, y, w, h = box
#         label = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
#         color = (0, 255, 0)
#         cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#         cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     # Show the frame
#     cv2.imshow("Object Detection", frame)

#     # Break on pressing 'q'
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()








# import cv2

# # Load the pre-trained MobileNet-SSD model and class names
# prototxt_path = "MobileNetSSD_deploy.prototxt"
# model_path = "MobileNetSSD_deploy.caffemodel"
# net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# # COCO class labels for detection
# CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
#            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
#            "dog", "horse", "motorbike", "person", "pottedplant",
#            "sheep", "sofa", "train", "tvmonitor"]

# # Start video capture (0 for default camera)
# cap = cv2.VideoCapture(0)

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Prepare input blob for the network
#     blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
#     net.setInput(blob)
#     detections = net.forward()

#     # Loop through detections
#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > 0.5:  # Confidence threshold
#             idx = int(detections[0, 0, i, 1])
#             label = CLASSES[idx]
#             box = detections[0, 0, i, 3:7] * \
#                 [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
#             (startX, startY, endX, endY) = box.astype("int")

#             # Draw bounding box and label
#             cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
#             cv2.putText(frame, f"{label}: {confidence:.2f}", (startX, startY-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Display the resulting frame
#     cv2.imshow("Object Detection - MobileNet SSD", frame)

#     # Break loop with 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release video capture and close windows
# cap.release()
# cv2.destroyAllWindows()












# import cv2

# # Path to your video file (correct the path)
# video_path = r"F:\Collage\vivek sis wed 12-12-24\VID-20241213-WA0002.mp4"

# # Open video file
# cap = cv2.VideoCapture(video_path)
# if not cap.isOpened():
#     print(f"Error: Could not open video file at {video_path}")
#     exit()

# # Load YOLO model (ensure config and weights paths are correct)
# config_path = "yolov3.cfg"
# weights_path = "yolov3.weights"
# labels_path = "coco.names"

# net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
# with open(labels_path, "r") as f:
#     labels = f.read().strip().split("\n")

# # Get the output layer names
# layer_names = net.getLayerNames()
# unconnected_out_layers = net.getUnconnectedOutLayers()

# # Check if unconnected_out_layers is a scalar or list
# if isinstance(unconnected_out_layers, int):
#     output_layer_names = [layer_names[unconnected_out_layers - 1]]
# else:
#     output_layer_names = [layer_names[i - 1] for i in unconnected_out_layers]

# print("Output Layer Names:", output_layer_names)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("End of video or failed to read frame.")
#         break

#     # YOLO object detection processing goes here

#     cv2.imshow("Object Detection Video", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
