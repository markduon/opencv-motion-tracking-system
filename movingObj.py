"""
movingObj.py

Description of the solution to Task Two:

In Task Two, we aim to detect and track pedestrians using a pre-trained MobileNet SSD detector provided by OpenCV's DNN module. The solution involves the following steps:

1. **Loading the Pre-trained Model and Class Names**
   - Read the class labels from object_detection_classes_coco.txt, which contains the labels for the COCO dataset.
   - This file maps class IDs to human-readable labels.
   - Use OpenCV's cv2.dnn.readNetFromTensorflow() to load the TensorFlow model (frozen_inference_graph.pb) and the configuration file (ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt).
   - This model is pre-trained on the COCO dataset and can detect various objects, including persons.

2. **Initializing Video Capture and Variables**
   - Use cv2.VideoCapture() to read the input video file specified by video_file.
   - Check if the video file is successfully opened.
   - frame_count: To keep track of the number of frames processed.
   - pedestrian_id: A counter to assign unique IDs to detected pedestrians.
   - trackers: A dictionary to store trackers for each pedestrian using their IDs.
   - max_disappeared: The maximum number of frames a tracker can be unmatched before it is removed.

3. **Processing Video Frames**
   - Loop through each frame of the video until the end.
   - Read each frame and increment frame_count.
   - Use the resize_image() function to resize frames to a standard VGA size (640x480) while maintaining the aspect ratio.
   - Resizing ensures consistent processing and improves performance.

4. **Detecting Pedestrians**
   - Convert the resized frame to a blob using cv2.dnn.blobFromImage().
   - Set the blob as input to the neural network using net.setInput(blob).
   - Run the forward pass using net.forward() to obtain detections.
   - For each detection: Check if it is a person with confidence above 0.35. Calculate bounding box coordinates, area, aspect ratio, and centroid. Filter out detections that are too large or have an unusual aspect ratio.

5. **Tracking Pedestrians**
   - Initialize unmatched trackers.
   - For each detected pedestrian: Attempt to match it with existing trackers based on centroid proximity. Update matched trackers with new detection data. Create new trackers for unmatched detections.
   - Update unmatched trackers by increasing their disappearance count.
   - Remove trackers that have disappeared for too long.

6. **Selecting Closest Pedestrians**
   - Sort tracked pedestrians by bounding box height (assuming taller ones are closer).
   - Select up to three closest pedestrians.

7. **Visualization**
   - Prepare frames showing: Original resized frame. Detections with green bounding boxes. Tracked pedestrians with blue bounding boxes and IDs. Closest pedestrians with red bounding boxes and IDs.
   - Display the frames in a 2x2 grid.

8. **Cleanup**
   - Release video capture and close all windows after processing.

"""


import cv2
import numpy as np
import sys


VGA_SIZE = (640, 480)  # VGA resolution
VGA_WIDTH, VGA_HEIGHT = VGA_SIZE

def resize_image(image, width=VGA_WIDTH, height=VGA_HEIGHT):
    """Resize the image while maintaining the aspect ratio."""
    aspect_ratio = image.shape[1] / image.shape[0]
    if aspect_ratio > width / height:
        new_width = width
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = height
        new_width = int(new_height * aspect_ratio)
    resized = cv2.resize(image, (new_width, new_height))
    return resized

def stack_frames(f1, f2, f3, f4):
    """Stack four frames into a single window."""
    top_row = cv2.hconcat([f1, f2])
    bottom_row = cv2.hconcat([f3, f4])
    return cv2.vconcat([top_row, bottom_row])

def task_one(video_file):
    """Task One: Background Modeling and Moving Object Extraction."""
    back_sub = cv2.createBackgroundSubtractorMOG2()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Unable to open video file:", video_file)
        sys.exit(1)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)
    
    print("Press 'q' to exit.")
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Unable to read the first frame.")
        sys.exit(1)

    first_frame = resize_image(first_frame)
    avg_frame = np.float32(first_frame)
    frame_number = 0

    while True:
        start_time = cv2.getTickCount()  # Start time of the frame processing
        
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        frame_resized = resize_image(frame)
        fg_mask = back_sub.apply(frame_resized)
        fg_mask_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask_clean = cv2.morphologyEx(fg_mask_clean, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_mask_clean, connectivity=8)
        min_area = 150
        num_persons = 0
        num_cars = 0
        num_others = 0

        for i in range(1, num_labels): 
            x, y, w, h, area = stats[i]
            if area >= min_area:
                aspect_ratio = w / h
                if 0.5 < aspect_ratio < 1.3:
                    num_persons += 1
                elif 1.3 <= aspect_ratio < 1.8:
                    num_cars += 1
                else:
                    num_others += 1

        cv2.rectangle(frame_resized, (10, 2), (100, 20), (255, 255, 255), -1)
        cv2.putText(frame_resized, str(frame_number), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        total_objects = num_persons + num_cars + num_others
        print(f"Frame {frame_number:04d}: {total_objects} objects ({num_persons} persons, {num_cars} cars, {num_others} others)")

        cv2.accumulateWeighted(frame_resized, avg_frame, 0.01)
        background_frame = cv2.convertScaleAbs(avg_frame)
        fg_mask_color = cv2.cvtColor(fg_mask_clean, cv2.COLOR_GRAY2BGR)
        moving_objects = cv2.bitwise_and(frame_resized, fg_mask_color)
        display_frame = stack_frames(frame_resized, background_frame, fg_mask_color, moving_objects)
        cv2.imshow('Moving Object Detection', display_frame)

        # Calculate elapsed time and adjust delay
        end_time = cv2.getTickCount()
        elapsed_time_ms = (end_time - start_time) / cv2.getTickFrequency() * 1000
        wait_time = max(1, delay - int(elapsed_time_ms))  # Ensure wait time is at least 1ms
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def task_two(video_file):
    """Task Two: Pedestrian Detection and Tracking with size filtering."""
    with open('object_detection_classes_coco.txt', 'r') as f:
        class_names = f.read().split('\n')

    net = cv2.dnn.readNetFromTensorflow(model='frozen_inference_graph.pb', config='ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt')
    cap = cv2.VideoCapture(video_file)

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)

    frame_count = 0
    pedestrian_id = 0
    trackers = {}
    max_disappeared = 5  

    while True:
        start_time = cv2.getTickCount()  # Start time of the frame processing
        
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_resized = resize_image(frame)
        frame_height, frame_width = frame_resized.shape[:2]
        blob = cv2.dnn.blobFromImage(frame_resized, size=(300, 300), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward()

        detected_pedestrians = []
        max_bbox_area = 0.3 * frame_width * frame_height

        for detection in detections[0, 0]:
            confidence = float(detection[2])
            class_id = int(detection[1])

            if confidence > 0.35 and class_names[class_id - 1] == 'person':  
                x_left_bottom = int(detection[3] * frame_width)
                y_left_bottom = int(detection[4] * frame_height)
                x_right_top = int(detection[5] * frame_width)
                y_right_top = int(detection[6] * frame_height)

                width = x_right_top - x_left_bottom
                height = y_right_top - y_left_bottom
                area = width * height
                aspect_ratio = width / height
                centroid = (int(x_left_bottom + width / 2), int(y_left_bottom + height / 2))

                if area < max_bbox_area and 0.3 < aspect_ratio < 0.7:
                    detected_pedestrians.append({
                        'bbox': (x_left_bottom, y_left_bottom, width, height),
                        'centroid': centroid,
                        'area': area,
                        'disappeared': 0,
                        'id': None
                    })

        unmatched_trackers = set(trackers.keys()) 

        for obj in detected_pedestrians:
            assigned = False
            for t_id, tracker in trackers.items():
                distance = np.linalg.norm(np.array(obj['centroid']) - np.array(tracker['centroid']))
                if distance < 50:
                    tracker['bbox'] = obj['bbox']
                    tracker['centroid'] = obj['centroid']
                    tracker['area'] = obj['area']
                    tracker['disappeared'] = 0  
                    obj['id'] = t_id
                    assigned = True
                    unmatched_trackers.discard(t_id)  
                    break

            if not assigned:
                pedestrian_id += 1
                obj['id'] = pedestrian_id
                trackers[pedestrian_id] = obj

        for t_id in unmatched_trackers:
            trackers[t_id]['disappeared'] += 1
            if trackers[t_id]['disappeared'] > max_disappeared:
                del trackers[t_id]

        tracked_pedestrians = list(trackers.values())
        tracked_pedestrians.sort(key=lambda x: x['bbox'][3], reverse=True)
        closest_pedestrians = tracked_pedestrians[:3]

        frame_with_detections = frame_resized.copy()
        frame_with_tracking = frame_resized.copy()
        frame_with_closest = frame_resized.copy()

        for obj in detected_pedestrians:
            x, y, w, h = obj['bbox']
            cv2.rectangle(frame_with_detections, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for t_id, tracker in trackers.items():
            x, y, w, h = tracker['bbox']
            cv2.rectangle(frame_with_tracking, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame_with_tracking, f'ID {t_id}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        for tracker in closest_pedestrians:
            x, y, w, h = tracker['bbox']
            cv2.rectangle(frame_with_closest, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame_with_closest, f'ID {tracker["id"]}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        top_row = np.hstack((frame_resized, frame_with_detections))
        bottom_row = np.hstack((frame_with_tracking, frame_with_closest))
        display_frame = np.vstack((top_row, bottom_row))

        cv2.imshow('Pedestrian Detection and Tracking', display_frame)

        # Calculate elapsed time and adjust delay
        end_time = cv2.getTickCount()
        elapsed_time_ms = (end_time - start_time) / cv2.getTickFrequency() * 1000
        wait_time = max(1, delay - int(elapsed_time_ms))
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main function to call the appropriate task based on command-line arguments."""
    if len(sys.argv) != 3:
        print("Usage: movingObj.py [-b|-d] <video_file>")
        print("Options:")
        print("  -b <video_file> : Perform Task One (Background Modelling)")
        print("  -d <video_file> : Perform Task Two (Pedestrian Detection and Tracking)")
        sys.exit(1)

    option = sys.argv[1]
    video_file = sys.argv[2]

    if option == '-b':
        task_one(video_file)
    elif option == '-d':
        task_two(video_file)
    else:
        print("Invalid option selected. Use -b for Task One or -d for Task Two.")
        sys.exit(1)

if __name__ == '__main__':
    main()
