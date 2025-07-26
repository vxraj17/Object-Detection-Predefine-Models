# import tkinter as tk
# from tkinter import filedialog, messagebox
# from PIL import Image, ImageTk
# import cv2
# from ultralytics import YOLO
# import threading
# import queue
# import time

# # Load the YOLO model
# model = YOLO('yolov8s.pt')

# class ObjectDetectionGUI:
#     def __init__(self, master):
#         self.master = master
#         master.title("Object Detection")

#         self.image_button = tk.Button(master, text="Capture and Detect Image", command=self.capture_and_detect_single_image)
#         self.image_button.pack(pady=10)

#         self.video_button = tk.Button(master, text="Video/Webcam Detection", command=self.run_video_detection)
#         self.video_button.pack(pady=10)

#         self.webcam_button = tk.Button(master, text="Real-time Image Capture", command=self.run_realtime_capture)
#         self.webcam_button.pack(pady=10)

#         self.cap = None
#         self.video_thread = None
#         self.image_queue = queue.Queue()
#         self.is_running_realtime = False
#         self.detection_window = None
#         self.detection_canvas = None
#         self.capture_window = None
#         self.capture_canvas = None
#         self.is_capturing = False

#     def capture_and_detect_single_image(self):
#         self.cap = cv2.VideoCapture(0)  # Open default webcam
#         if not self.cap.isOpened():
#             messagebox.showerror("Error", "Could not open webcam.")
#             return

#         ret, frame = self.cap.read()
#         self.cap.release()  # Release the webcam immediately after capturing

#         if ret:
#             # Perform detection on the captured frame
#             results = model.predict(frame, verbose=False, conf=0.5)
#             image_with_detections = frame.copy()  # Create a copy to draw on

#             for result in results:
#                 boxes = result.boxes.cpu().numpy()
#                 confidence_scores = boxes.conf
#                 class_ids = boxes.cls.astype(int)
#                 class_names = result.names

#                 for i in range(len(boxes)):
#                     x1, y1, x2, y2 = map(int, boxes.xyxy[i])
#                     confidence = confidence_scores[i]
#                     class_id = class_ids[i]
#                     class_name = class_names[class_id]
#                     label = f'{class_name}: {confidence:.2f}'
#                     color = (0, 255, 0)
#                     cv2.rectangle(image_with_detections, (x1, y1), (x2, y2), color, 2)
#                     cv2.putText(image_with_detections, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#             # Display the image with detections
#             image_rgb = cv2.cvtColor(image_with_detections, cv2.COLOR_BGR2RGB)
#             img = Image.fromarray(image_rgb)
#             img_tk = ImageTk.PhotoImage(img)

#             if not self.detection_window:
#                 self.detection_window = tk.Toplevel(self.master)
#                 self.detection_window.title("Captured Image Detection")
#                 self.detection_canvas = tk.Canvas(self.detection_window)
#                 self.detection_canvas.pack()

#             self.detection_canvas.config(width=img_tk.width(), height=img_tk.height())
#             self.detection_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
#             self.detection_canvas.image = img_tk

#         else:
#             messagebox.showerror("Error", "Could not capture frame from webcam.")

#     def run_video_detection(self):
#         file_path = filedialog.askopenfilename(
#             title="Select Video File",
#             filetypes=[("Video files", "*.mp4;*.avi;*.mov")]
#         )
#         if file_path:
#             self.open_video_window(file_path)
#         else:
#             self.open_video_window(0)

#     def open_video_window(self, video_source):
#         self.detection_window = tk.Toplevel(self.master)
#         self.detection_window.title("Video/Webcam Detection")
#         self.detection_canvas = tk.Canvas(self.detection_window)
#         self.detection_canvas.pack()

#         self.cap = cv2.VideoCapture(video_source)
#         if not self.cap.isOpened():
#             messagebox.showerror("Error", "Could not open video source.")
#             return

#         self.is_running_realtime = True
#         self.video_thread = threading.Thread(target=self.update_frame)
#         self.video_thread.daemon = True
#         self.video_thread.start()

#         self.detection_window.protocol("WM_DELETE_WINDOW", self.stop_video)

#     def update_frame(self):
#         while self.is_running_realtime:
#             ret, frame = self.cap.read()
#             if not ret:
#                 self.stop_video()
#                 break

#             results = model.predict(frame, verbose=False, conf=0.5)
#             for result in results:
#                 boxes = result.boxes.cpu().numpy()
#                 confidence_scores = boxes.conf
#                 class_ids = boxes.cls.astype(int)
#                 class_names = result.names

#                 for i in range(len(boxes)):
#                     x1, y1, x2, y2 = map(int, boxes.xyxy[i])
#                     confidence = confidence_scores[i]
#                     class_id = class_ids[i]
#                     class_name = class_names[class_id]
#                     label = f'{class_name}: {confidence:.2f}'
#                     color = (0, 255, 0)
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             img = Image.fromarray(frame_rgb)
#             img_tk = ImageTk.PhotoImage(img)

#             if self.detection_canvas:
#                 self.detection_canvas.config(width=img_tk.width(), height=img_tk.height())
#                 self.detection_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
#                 self.detection_canvas.image = img_tk

#             time.sleep(0.03)

#     def stop_video(self):
#         self.is_running_realtime = False
#         if self.cap and self.cap.isOpened():
#             self.cap.release()
#         if self.detection_window:
#             self.detection_window.destroy()
#             self.detection_window = None
#             self.detection_canvas = None

#     def run_realtime_capture(self):
#         self.open_realtime_capture_window()

#     def open_realtime_capture_window(self):
#         self.capture_window = tk.Toplevel(self.master)
#         self.capture_window.title("Real-time Image Capture")
#         self.capture_canvas = tk.Canvas(self.capture_window)
#         self.capture_canvas.pack()
#         self.capture_button = tk.Button(self.capture_window, text="Capture Image", command=self.capture_and_detect)
#         self.capture_button.pack(pady=5)

#         self.cap = cv2.VideoCapture(0)
#         if not self.cap.isOpened():
#             messagebox.showerror("Error", "Could not open webcam.")
#             return

#         self.is_capturing = True
#         self.capture_thread = threading.Thread(target=self.update_capture_frame)
#         self.capture_thread.daemon = True
#         self.capture_thread.start()

#         self.capture_window.protocol("WM_DELETE_WINDOW", self.stop_capture)

#     def update_capture_frame(self):
#         while self.is_capturing:
#             ret, frame = self.cap.read()
#             if not ret:
#                 self.stop_capture()
#                 break

#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             img = Image.fromarray(frame_rgb)
#             img_tk = ImageTk.PhotoImage(img)

#             if self.capture_canvas:
#                 self.capture_canvas.config(width=img_tk.width(), height=img_tk.height())
#                 self.capture_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
#                 self.capture_canvas.image = img_tk

#             time.sleep(0.03)

#     def capture_and_detect(self):
#         if self.cap and self.cap.isOpened():
#             ret, frame = self.cap.read()
#             if ret:
#                 captured_image_path = "captured_image.png" # You can skip saving if you want
#                 cv2.imwrite(captured_image_path, frame)
#                 self.detect_and_show_image(captured_image_path)
#             else:
#                 messagebox.showerror("Error", "Could not capture frame.")

#     def stop_capture(self):
#         self.is_capturing = False
#         if self.cap and self.cap.isOpened():
#             self.cap.release()
#         if self.capture_window:
#             self.capture_window.destroy()
#             self.capture_window = None
#             self.capture_canvas = None

# root = tk.Tk()
# gui = ObjectDetectionGUI(root)
# root.mainloop()








import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import threading
import queue
import time

# Load the YOLO model (you can choose a different one for accuracy)
model = YOLO('yolov8x.pt')

class ObjectDetectionGUI:
    def __init__(self, master):
        self.master = master
        master.title("Object Detection")

        self.image_button = tk.Button(master, text="Image Detection", command=self.run_image_detection)
        self.image_button.pack(pady=10)

        self.video_button = tk.Button(master, text="Video/Webcam Detection", command=self.run_video_detection)
        self.video_button.pack(pady=10)

        self.webcam_button = tk.Button(master, text="Real-time Image Capture", command=self.run_realtime_capture)
        self.webcam_button.pack(pady=10)

        self.cap = None  # VideoCapture object for webcam
        self.video_thread = None
        self.image_queue = queue.Queue()
        self.is_running_realtime = False
        self.detection_window = None
        self.detection_canvas = None

    def run_image_detection(self):
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg")]
        )
        if file_path:
            self.detect_and_show_image(file_path)

    def detect_and_show_image(self, image_path):
        try:
            results = model.predict(image_path, verbose=False, conf=0.5)
            image = cv2.imread(image_path)
            if image is None:
                messagebox.showerror("Error", "Could not open or find the image.")
                return

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
                    color = (0, 255, 0)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(image_rgb)
            img_tk = ImageTk.PhotoImage(img)

            if not self.detection_window:
                self.detection_window = tk.Toplevel(self.master)
                self.detection_window.title("Image Detection Result")
                self.detection_canvas = tk.Canvas(self.detection_window)
                self.detection_canvas.pack()

            self.detection_canvas.config(width=img_tk.width(), height=img_tk.height())
            self.detection_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.detection_canvas.image = img_tk  # Keep a reference!

        except FileNotFoundError:
            messagebox.showerror("Error", f"Image not found at '{image_path}'")
        except Exception as e:
            messagebox.showerror("Error", f"Error during image detection: {e}")

    def run_video_detection(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4;*.avi;*.mov")]
        )
        if file_path:
            self.open_video_window(file_path)
        else:
            # If no file selected, assume webcam (index 0)
            self.open_video_window(0)

    def open_video_window(self, video_source):
        self.detection_window = tk.Toplevel(self.master)
        self.detection_window.title("Video/Webcam Detection")
        self.detection_canvas = tk.Canvas(self.detection_window)
        self.detection_canvas.pack()

        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open video source.")
            return

        self.is_running_realtime = True
        self.video_thread = threading.Thread(target=self.update_frame)
        self.video_thread.daemon = True
        self.video_thread.start()

        self.detection_window.protocol("WM_DELETE_WINDOW", self.stop_video)

    def update_frame(self):
        while self.is_running_realtime:
            ret, frame = self.cap.read()
            if not ret:
                self.stop_video()
                break

            results = model.predict(frame, verbose=False, conf=0.5)
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
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(img)

            if self.detection_canvas:
                self.detection_canvas.config(width=img_tk.width(), height=img_tk.height())
                self.detection_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                self.detection_canvas.image = img_tk  # Keep a reference!

            time.sleep(0.03) # Adjust for desired frame rate

    def stop_video(self):
        self.is_running_realtime = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.detection_window:
            self.detection_window.destroy()
            self.detection_window = None
            self.detection_canvas = None

    def run_realtime_capture(self):
        self.open_realtime_capture_window()

    def open_realtime_capture_window(self):
        self.capture_window = tk.Toplevel(self.master)
        self.capture_window.title("Real-time Image Capture")
        self.capture_canvas = tk.Canvas(self.capture_window)
        self.capture_canvas.pack()
        self.capture_button = tk.Button(self.capture_window, text="Capture Image", command=self.capture_and_detect)
        self.capture_button.pack(pady=5)

        self.cap = cv2.VideoCapture(0) # Open default webcam
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            return

        self.is_capturing = True
        self.capture_thread = threading.Thread(target=self.update_capture_frame)
        self.capture_thread.daemon = True
        self.capture_thread.start()

        self.capture_window.protocol("WM_DELETE_WINDOW", self.stop_capture)

    def update_capture_frame(self):
        while self.is_capturing:
            ret, frame = self.cap.read()
            if not ret:
                self.stop_capture()
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(img)

            if self.capture_canvas:
                self.capture_canvas.config(width=img_tk.width(), height=img_tk.height())
                self.capture_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                self.capture_canvas.image = img_tk

            time.sleep(0.03)

    def capture_and_detect(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Save the captured frame (optional)
                captured_image_path = "captured_image.png"
                cv2.imwrite(captured_image_path, frame)
                self.detect_and_show_image(captured_image_path)
            else:
                messagebox.showerror("Error", "Could not capture frame.")

    def stop_capture(self):
        self.is_capturing = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.capture_window:
            self.capture_window.destroy()
            self.capture_window = None
            self.capture_canvas = None

root = tk.Tk()
gui = ObjectDetectionGUI(root)
root.mainloop()


