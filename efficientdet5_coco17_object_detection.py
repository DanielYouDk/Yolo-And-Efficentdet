# how to run the code 
# python efficientdet5_coco17_object_detection.py --image images/(your-image.jpg) --model (efficientdet5_model) in command prompt
# note : the stuff around parentheses are placeholders and should be correctly replaced.
# this command prompt assumes you have a seperate folder called images that stores all the images to test on the Efficientdet5 model.
# make sure you are in the same directory as your python file

# packages
import numpy as np
import argparse
import time
import cv2
import tensorflow as tf

# Load label mapping
label_map_path = "Label_mapping/mscoco_label_map.pbtxt"
label_map = {}
with open(label_map_path, 'r') as f:
    lines = f.readlines()
    class_id = None
    display_name = None
    for line in lines:
        if "id:" in line:
            class_id = int(line.split(":")[-1])
        elif "display_name:" in line:
            display_name = line.split(":")[-1].strip().replace("'", "")
        if class_id is not None and display_name is not None:
            label_map[class_id] = display_name
            class_id = None
            display_name = None

# specify the input image and the base path to the EfficientDet model through command-line arguments, along with optional
# confidence and threshold values.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-m", "--model", required=True,
                help="path to EfficientDet model directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# load EfficientDet model
model_path = args["model"]
model = tf.saved_model.load(model_path)

# read and preprocess the input image
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]
input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
input_image = tf.image.resize(input_image, (512, 512))
input_image = tf.expand_dims(input_image, 0)
input_image = tf.cast(input_image, dtype=tf.uint8)

# make prediction on the input image
start = time.time()
detections = model(input_image)
end = time.time()

# processes the detections obtained from the EfficientDet model by filtering out weak
# predictions based on the specified confidence threshold and draws bounding boxes and labels for the detected objects on the image.
for i in range(len(detections['detection_boxes'][0])):
    confidence = detections['detection_scores'][0][i].numpy()
    if confidence > args["confidence"]:
        classID = int(detections['detection_classes'][0][i].numpy())
        label = label_map.get(classID, 'Unknown')  # Get the label name from the mapping or 'Unknown' if not found
        box = detections['detection_boxes'][0][i].numpy() * np.array([H, W, H, W])
        (startY, startX, endY, endX) = box.astype("int")

        # draw the prediction on the image
        label_text = "{}: {:.2f}%".format(label, confidence * 100)
        color = [int(c) for c in np.random.randint(0, 255, size=3)]
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label_text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# the processed image with bounding boxes and labels is displayed
cv2.imshow("EfficientDet Object Detection", image)
cv2.waitKey(0)
