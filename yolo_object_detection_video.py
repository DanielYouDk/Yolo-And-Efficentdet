# how to run code in cmd
# python yolo_object_detection_video.py -i videos/your-video.mp4 -o outputs/video-output.avi -y yolov4-info -c # -t #
# Note : the threshold and confidence values don't have to be defined as they have a default value that is pretty much already optimized
# make sure you are in the same directory as your python file

# Required packages
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

def parse_cmd_args():
    """
    Parse command line arguments.

    Returns:
        dict: Dictionary containing the parsed command line arguments.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_video", required=True)
    ap.add_argument("-o", "--output_video", required=True)
    ap.add_argument("-y", "--yolo_base", required=True)
    ap.add_argument("-c", "--confidence_threshold", type=float, default=0.5)
    ap.add_argument("-t", "--threshold", type=float, default=0.3)
    return vars(ap.parse_args())

def load_yolo_labels(yolo_base_path):
    """
    Load the labels from the COCO dataset.

    Args:
        yolo_base_path (str): Base path to YOLO directory.

    Returns:
        list: List of labels.
    """
    labels_file_path = os.path.join(yolo_base_path, "coco.names")
    with open(labels_file_path, "r") as f:
        labels = [line.strip() for line in f]
    return labels

def initialize_random_colors(num_labels):
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

def load_yolo_model(yolo_base_path):
    """
    Load the pre-trained YOLO model from disk.

    Args:
        yolo_base_path (str): Base path to YOLO directory.

    Returns:
        tuple: Tuple containing the YOLO model and its layer names.
    """
    weights_file_path = os.path.join(yolo_base_path, "yolov4.weights")
    config_file_path = os.path.join(yolo_base_path, "yolov4.cfg")
    net = cv2.dnn.readNetFromDarknet(config_file_path, weights_file_path)
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, ln

def perform_object_detection(yolo_net, yolo_layers, frame, frame_width, frame_height, confidence_threshold):
    """
    Perform object detection on the given frame using YOLO.

    Args:
        yolo_net (cv2.dnn_Net): YOLO model.
        yolo_layers (list): List of YOLO layer names.
        frame (numpy.ndarray): Input frame.
        frame_width (int): Width of the frame.
        frame_height (int): Height of the frame.
        confidence_threshold (float): Minimum probability to filter weak detections.

    Returns:
        tuple: Tuple containing lists of boxes, confidences, and class IDs.
    """
    def get_object_info(detection, frame_width, frame_height):
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > confidence_threshold:
            box = detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
            (center_x, center_y, width, height) = box.astype("int")
            x = int(center_x - (width / 2))
            y = int(center_y - (height / 2))
            return [x, y, int(width), int(height)], float(confidence), class_id
        return None

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    layer_outputs = yolo_net.forward(yolo_layers)

    object_info = [
        get_object_info(detection, frame_width, frame_height)
        for output in layer_outputs
        for detection in output
    ]

    boxes = [info[0] for info in object_info if info is not None]
    confidences = [info[1] for info in object_info if info is not None]
    class_ids = [info[2] for info in object_info if info is not None]

    return boxes, confidences, class_ids


def main():
    # Parse command line arguments
    args = parse_cmd_args()

    # Validate input video file
    if not os.path.isfile(args["input_video"]):
        print("Error: Input video file not found.")
        return

    # Validate output directory
    output_directory = os.path.dirname(args["output_video"])
    if not os.access(output_directory, os.W_OK):
        print("Error: Output directory not accessible for writing.")
        return

    # Load YOLO labels and initialize random colors
    yolo_labels = load_yolo_labels(args["yolo_base"])
    yolo_colors = initialize_random_colors(len(yolo_labels))

    # Load the YOLO model
    yolo_net, yolo_layers = load_yolo_model(args["yolo_base"])

    # Open the input video
    video_stream = cv2.VideoCapture(args["input_video"])

    # Get the resolution of the video
    frame_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("The video resolution is {}x{}".format(frame_width, frame_height))

    # Get the total number of frames in the video
    total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    print("The video has {} frames.".format(total_frames))

    # Initialize the video writer and other helpful variables
    video_writer = None
    elapsed_time, frames_processed = 0, 0

    while True:
        # Read a frame from the video
        (frame_obtained, current_frame) = video_stream.read()
        if not frame_obtained:
            break

        # If the frame dimensions are not yet set, set them
        if frame_width is None or frame_height is None:
            frame_height, frame_width = current_frame.shape[:2]

        # Perform object detection on the frame using YOLO
        start_time = time.time()
        boxes, confidences, class_ids = perform_object_detection(yolo_net, yolo_layers, current_frame, frame_width, frame_height, args["confidence_threshold"])
        end_time = time.time()

        # Calculate the time taken to process this frame
        elapsed_time += (end_time - start_time)
        frames_processed += 1

        # Apply non-maxima suppression to filter out overlapping detections
        indices = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence_threshold"], args["threshold"])

        # Print information about the detection results
        print("Detected objects in frame {}: {}".format(frames_processed, len(indices)))

        # Print progress information
        print("Processing frame {}/{}".format(frames_processed, total_frames))

        # Print frame processing time
        print("Frame processing time: {:.4f} seconds".format(end_time - start_time))

        if len(indices) > 0:
            object_info = [(boxes[i], yolo_colors[class_ids[i]], yolo_labels[class_ids[i]], confidences[i]) for i in indices.flatten()]

            for box, color, label, confidence in object_info:
                x, y, w, h = box
                color = [int(c) for c in color]
                cv2.rectangle(current_frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(label, confidence)

                # set the font size of the text
                font_scale = 0.9

                # Set the text color
                text_color = (0, 0, 0)

                cv2.putText(current_frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, float(font_scale), color, 2)

        if video_writer is None:
            # Create the video writer
            video_writer = cv2.VideoWriter(args["output_video"], VIDEO_FOURCC, VIDEO_FPS, (frame_width, frame_height), True)
            # Calculate the average processing time per frame and total time remaining to finish the video
            if total_frames > 0:
                avg_processing_time_per_frame = elapsed_time / frames_processed
                total_time_remaining = (total_frames - frames_processed) * avg_processing_time_per_frame
                print("Average processing time per frame: {:.4f} seconds".format(avg_processing_time_per_frame))
                print("Total time to finish in seconds: {:.4f}".format(total_time_remaining))

        # Write the processed frame to the output video
        video_writer.write(current_frame)

    # Release resources and print completion message
    video_writer.release()
    video_stream.release()
    print("Complete")

    # Open the output video automatically after processing
    try:
        print("Opening the output video...")
        subprocess.run(['start', args["output_video"]], shell=True)
    except Exception as e:
        print("Error opening the output video: {}".format(e))


if __name__ == "__main__":
    main()


