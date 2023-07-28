# how to run code 
# python yolo_object_detection_video.py -i videos/(your-video.mp4) -o outputs/(video-output.avi) -y yolo-info -c # -t # in command prompt
# everything in parentheses are placeholders and should be replaced accordingly
# this command prompt assumes you have two seperate folders, videos and outputs to store the video files and the video files returned as an 
# output after ran in the code 
# Note : the threshold and confidence values don't have to be defined as they have a default value that was already tested and optimized
# make sure you are in the same directory as your python file

# packages
import numpy as np
import argparse
import cv2
import os
import time
import subprocess

def parse_arguments():
    # Parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="path to input video")
    ap.add_argument("-o", "--output", required=True, help="path to output video")
    ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
    return vars(ap.parse_args())

def load_labels(yolo_path):
    # Load the labels from the COCO dataset
    labelsPath = os.path.sep.join([yolo_path, "coco.names"])
    return open(labelsPath).read().strip().split("\n")

def initialize_colors(num_labels):
    # Initialize random colors for each label
    np.random.seed(42)
    return np.random.randint(0, 255, size=(num_labels, 3), dtype="uint8")

def load_yolo_model(yolo_path):
    # Load the pre-trained YOLO model from disk
    weightsPath = os.path.sep.join([yolo_path, "yolov3.weights"])
    configPath = os.path.sep.join([yolo_path, "yolov3.cfg"])
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, ln

def detect_objects(net, ln, frame, frame_width, frame_height, confidence_threshold):
    # Perform object detection on the given frame using YOLO
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(ln)
    boxes = []
    confidences = []
    classIDs = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    return boxes, confidences, classIDs

def main():
    # Parse command line arguments
    args = parse_arguments()

    # Load YOLO labels and initialize random colors
    LABELS = load_labels(args["yolo"])
    COLORS = initialize_colors(len(LABELS))

    # Load the YOLO model
    net, ln = load_yolo_model(args["yolo"])

    # Open the input video
    vs = cv2.VideoCapture(args["input"])

    # Get the resolution of the video
    frame_width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("The video resolution is {}x{}".format(frame_width, frame_height))

    # Get the total number of frames in the video
    total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT)) if cv2.__version__.startswith('2') else int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    print("The video has {} frames.".format(total_frames))

    # Initialize the video writer and other helpful variables
    writer = None
    frame_width, frame_height = None, None
    elapsed_time, frames_processed = 0, 0

    while True:
        # Read a frame from the video
        (grabbed, frame) = vs.read()
        if not grabbed:
            break

        # If the frame dimensions are not yet set, set them
        if frame_width is None or frame_height is None:
            frame_height, frame_width = frame.shape[:2]

        # Perform object detection on the frame using YOLO
        start_time = time.time()
        boxes, confidences, classIDs = detect_objects(net, ln, frame, frame_width, frame_height, args["confidence"])
        end_time = time.time()

        # Calculate the time taken to process this frame
        elapsed_time += (end_time - start_time)
        frames_processed += 1

        # Apply non-maxima suppression to filter out overlapping detections
        indices = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

        # Print information about the detection results
        print("Detected objects in frame {}: {}".format(frames_processed, len(indices)))

        # Print progress information
        print("Processing frame {}/{}".format(frames_processed, total_frames))

        # Print frame processing time
        print("Frame processing time: {:.4f} seconds".format(end_time - start_time))

        # Draw bounding boxes and labels for the detected objects
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if writer is None:
            # Specify the desired output video file type here.
            # Some common formats that are likely to work without errors include:
            # - "mp4v" for MP4 format
            # - "XVID" for AVI format
            # - "MJPG" for Motion JPEG format (each frame is treated as a separate JPEG image)
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame_width, frame_height), True)
            # Calculate the average processing time per frame and total time remaining to finish the video
            if total_frames > 0:
                avg_processing_time_per_frame = elapsed_time / frames_processed
                total_time_remaining = (total_frames - frames_processed) * avg_processing_time_per_frame
                print("Average processing time per frame: {:.4f} seconds".format(avg_processing_time_per_frame))
                print("Total time to finish in seconds: {:.4f}".format(total_time_remaining))

        # Write the processed frame to the output video
        writer.write(frame)

    # Release resources and print completion message
    print("Finishing...")
    writer.release()
    vs.release()
    print("Complete")
    
    # Open the output video automatically after processing
    try:
        print("Opening the output video...")
        subprocess.run(['start', args["output"]], shell=True)
    except Exception as e:
        print("Error opening the output video: {}".format(e))


if __name__ == "__main__":
    main()

