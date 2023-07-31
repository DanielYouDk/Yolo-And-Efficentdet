
Accessing the TensorFlow Model Zoo:
To access the TensorFlow model zoo page, open your web browser and click on the following link: 
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

Downloading EfficientDet Models:
On the TensorFlow model zoo page, scroll down to find the section for "EfficientDet D7 1536x1536" and "EfficientDet D5 1280x1280" models. Download the tar.gz executable for both models by clicking on the provided links.

Extracting Required Files:
Once the download is complete, extract the downloaded tar.gz files. You'll find two folders named "assets" and "variables," along with the "saved_model.pb" file. Create a new separate folder to store all these files. Take note of the name and location of this folder on your computer.

Updating Command Line Parameters:
In the code file named "efficientdet5_coco17_object_detection," update the command line parameter "--model" with the name of the separate folder you created for the EfficientDet D5 model. This ensures that the code uses the correct model for object detection even if you named the folder differently than the example shows. The same applies to the code file "efficientdet7_coco17_object_detection."

Downloading Pretrained YOLOV4 Model Weights:
Click on the following link to download the pretrained YOLOV4 model weights file: YOLOV4 Model Weights

https://drive.google.com/file/d/1Hwygr_j2DAMHpDTfhi0lPlQ9OD2Auyrg/view?usp=sharing

Placing YOLOV4 Model Weights:
After downloading the YOLOV4 model weights file, place the downloaded weights file inside the folder "yolov4-info". This folder will be used in the code for loading the YOLO Model.

Reading Code File Instructions:
Before running the code, make sure to read the instructions provided at the top of each code file. This will help you understand how to properly execute the code and utilize the downloaded models and weights.
