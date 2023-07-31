Click on the link below to access the TensorFlow model zoo page:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

Once on the TensorFlow model zoo page, find the section for "EfficientDet D7 1536x1536" and "EfficientDet D5 1280x1280" model and download the tar.gz executable.

After its done downloading, the files and folders you need to extract are the assets and variable folders including the saved_model.pb file. Make a seperate folder to store all of these and make sure to note its name/location on your computer.

In the code file named "efficientdet5_coco17_object_detection," you need to update the command line parameter "--model" with the seperate folder name of the EfficientDet D5 model you just made. This will ensure that the code uses the correct model for the object detection if you don't name it the same as the example shows. This also applies to the code file named "efficientdet7_coco17_object_detection"

Additionally, there is another link to download pretrained weights used to train the YOLOV4 Model. Click on the link below to download the file:
https://drive.google.com/file/d/1Hwygr_j2DAMHpDTfhi0lPlQ9OD2Auyrg/view?usp=sharing

Once downloaded, place the pretrained YOLOV4 model weights file in the "yolo-info4" folder. This folder is used in the code for loading the YOLO Model.

Before running the code, make sure to read the instructions provided at the top of each code file to understand how to execute the code properly.
