import glob
import cv2 as cv
import numpy as np

net = cv.dnn.readNet('yolov3.weights', 'yolov3.cfg')  # load the networks
classes = []

with open('coco.names', 'r') as f:      # load all the python programs
    classes = f.read().splitlines()

path = glob.glob("C:/Users/Px/PycharmProjects/MAJOR/Images/*.*")


for file in path:
    print(file)

    image = cv.imread(file)
    cv.imshow('Original', image)

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    gaussian_blur = cv.GaussianBlur(image,(11, 11),cv.BORDER_DEFAULT)
    cv.imshow('gaussian Filter', gaussian_blur)


    height, width, _ = image.shape
    blob = cv.dnn.blobFromImage(image, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)  # Return input image after the mean subtraction, normalizing, channel swapping


    net.setInput(blob)  # set input from blob into the networks
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)
    boxes = []  # extract bounding boxes
    confidences = []
    class_ids = []
    for output in layerOutputs:  # extract all the inormation from the layerOutput
        for detection in output:  # extract information from each of the for loops
            scores = detection[5:]  # array scores used to store process predictions
            class_id = np.argmax(scores)  # find locations that contains the higher scores
            confidence = scores[class_id]  # extraxt the higher in and then assign into the confidence
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    print(len(boxes))

    # non max suppression

    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # 0.4-> max suppression
    # print(indexes.flatten())

    font = cv.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i]))
        color = colors[i]
        cv.rectangle(image, (x, y), (x + w, y + h), color, 2)  # put rectangle on the detected object
        cv.putText(image, label + "" + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)  # put the label

    cv.imshow('Detected Objects', image)
    cv.waitKey(0)
    cv.destroyAllWindows()











