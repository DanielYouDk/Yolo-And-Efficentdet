# how to run code in cmd
# python yolo_object_detection_video.py -i videos/your-video.mp4 -o outputs/video-output.avi -y yolo-info -c # -t #
# example of cmd code :
# python yolo_object_detection_video.py -i videos/runningdog.mp4 -o outputs/runningdog-output.avi -y yolo-info -c 0.4 -t 0.2
# Note : the threshold and confidence values don't have to be defined as they have a default value that is pretty much already optimized
# make sure you are in the same directory as your python file

# packages
import numpy as np
import argparse
import cv2
import os
import time
import subprocess
import colorsys

# Constants for video writer parameters
VIDEO_FOURCC = cv2.VideoWriter_fourcc(*"MJPG")
VIDEO_FPS = 30

def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        dict: Dictionary containing the parsed command line arguments.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="path to input video")
    ap.add_argument("-o", "--output", required=True, help="path to output video")
    ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
    return vars(ap.parse_args())

def load_labels(yolo_path):
    """
    Load the labels from the COCO dataset.

    Args:
        yolo_path (str): Base path to YOLO directory.

    Returns:
        list: List of labels.
    """
    labels_path = os.path.join(yolo_path, "coco.names")
    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f]
    return labels

def initialize_colors(num_labels):
    """
    Initialize random colors for each label.

    Args:
        num_labels (int): Number of labels.

    Returns:
        numpy.ndarray: Array of random colors.
    """
    hsv_colors = [(x / num_labels, 1.0, 1.0) for x in range(num_labels)]
    rgb_colors = [colorsys.hsv_to_rgb(*color) for color in hsv_colors]
    colors = (np.array(rgb_colors) * 255).astype(np.uint8)
    return colors

def load_yolo_model(yolo_path):
    """
    Load the pre-trained YOLO model from disk.

    Args:
        yolo_path (str): Base path to YOLO directory.

    Returns:
        tuple: Tuple containing the YOLO model and its layer names.
    """
    weights_path = os.path.join(yolo_path, "yolov3.weights")
    config_path = os.path.join(yolo_path, "yolov3.cfg")
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, ln

def perform_object_detection(net, ln, frame, frame_width, frame_height, confidence_threshold):
    """
    Perform object detection on the given frame using YOLO.

    Args:
        net (cv2.dnn_Net): YOLO model.
        ln (list): List of YOLO layer names.
        frame (numpy.ndarray): Input frame.
        frame_width (int): Width of the frame.
        frame_height (int): Height of the frame.
        confidence_threshold (float): Minimum probability to filter weak detections.

    Returns:
        tuple: Tuple containing lists of boxes, confidences, and class IDs.
    """
    def get_object_info(detection, frame_width, frame_height):
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > confidence_threshold:
            box = detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            return [x, y, int(width), int(height)], float(confidence), classID
        return None

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(ln)

    object_info = [
        get_object_info(detection, frame_width, frame_height)
        for output in layer_outputs
        for detection in output
    ]

    boxes = [info[0] for info in object_info if info is not None]
    confidences = [info[1] for info in object_info if info is not None]
    classIDs = [info[2] for info in object_info if info is not None]

    return boxes, confidences, classIDs


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Validate input video file
    if not os.path.isfile(args["input"]):
        print("Error: Input video file not found.")
        return

    # Validate output directory
    output_dir = os.path.dirname(args["output"])
    if not os.access(output_dir, os.W_OK):
        print("Error: Output directory not accessible for writing.")
        return

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
        boxes, confidences, classIDs = perform_object_detection(net, ln, frame, frame_width, frame_height, args["confidence"])
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

        if len(indices) > 0:
            object_info = [(boxes[i], COLORS[classIDs[i]], LABELS[classIDs[i]], confidences[i])
                           for i in indices.flatten()]

            for box, color, label, confidence in object_info:
                x, y, w, h = box
                color = [int(c) for c in color]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(label, confidence)

                # Increase the font size of the text
                font_scale = 1.3

                # Set the text color
                text_color = (0, 0, 0)

                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, float(font_scale), text_color, 4)
                
        if writer is None:
            # Create the video writer
            writer = cv2.VideoWriter(args["output"], VIDEO_FOURCC, VIDEO_FPS, (frame_width, frame_height), True)
            # Calculate the average processing time per frame and total time remaining to finish the video
            if total_frames > 0:
                avg_processing_time_per_frame = elapsed_time / frames_processed
                total_time_remaining = (total_frames - frames_processed) * avg_processing_time_per_frame
                print("Average processing time per frame: {:.4f} seconds".format(avg_processing_time_per_frame))
                print("Total time to finish in seconds: {:.4f}".format(total_time_remaining))

        # Write the processed frame to the output video
        writer.write(frame)

    # Release resources and print completion message
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

