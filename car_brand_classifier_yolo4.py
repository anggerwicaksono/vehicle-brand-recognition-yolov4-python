# Copyright © 2020 by Spectrico
# Licensed under the MIT License
# Based on the tutorial by Adrian Rosebrock: https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
# Usage: $ python car_brand_classifier_yolo4.py --image cars.jpg

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
import color_classifier
import make_classifier
import streamlit as st
import urllib

st.sidebar.header('Features Setting')
values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
def_con = values.index(0.5)
def_tres = values.index(0.3)
con = st.sidebar.selectbox('Confidence Value (default 0.5)', values, index=def_con)
tres = st.sidebar.selectbox('Treshold Value (default 0.3)', values, index=def_tres)


def main():
    # Render the readme as markdown using st.markdown.
    st.write("""
    # CCD Car Color Detection
    Implementation of AI Object Detection using YOLOv4 (OpenCV2 DNN backend) on Streamlit
    (code forked from https://github.com/spectrico/car-color-classifier-yolo4-python)
    
    """)

    # Download external dependencies.
    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)

    # Once we have the dependencies, add a selector for the app mode on the sidebar.

    run_the_app()

def download_file(file_path):
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(file_path):
        if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
            return
        elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
            return

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % file_path)
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
                        (file_path, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

def run_the_app():
    uf = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    st.sidebar.write("""
    [[Open in Github]](https://github.com/anggerwicaksono/car-color-classifier-yolo4-python.git)
    """)
    ap = argparse.ArgumentParser()
    ap.add_argument("-y", "--yolo", default='yolov4', help="base path to YOLO directory")
    args = vars(ap.parse_args())

    if uf is not None:
        st.image(uf, caption="Uploaded Image", use_column_width=True)
        with open(uf.name, 'wb') as f:
            f.write(uf.read())
            st.write("Processing Image ...")

    car_color_classifier = color_classifier.Classifier()
    car_make_classifier = make_classifier.Classifier()

    labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    #configPath = os.path.sep.join([args["yolo"], "yolov4.cfg"])
    def load_network(configPath, weightsPath):
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        output_layer_names = net.getLayerNames()
        output_layer_names = [output_layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return net, output_layer_names

    net, output_layer_names = load_network("yolov4.cfg", "yolov4.weights")


# load our input image and grab its spatial dimensions
    if uf is not None:
        image = cv2.imread(uf.name)
        (H, W) = image.shape[:2]

        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (608, 608), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)
    
        boxes = []
        confidences = []
        classIDs = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > con:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, con, tres)
        start = time.time()
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                if classIDs[i] == 2 or classIDs[i] == 7:
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    resultC = car_color_classifier.predict(image[max(y, 0):y + h, max(x, 0):x + w])
                    resultM = car_make_classifier.predict(image[max(y, 0):y + h, max(x, 0):x + w])
                    textC = "{}: {:.4f}".format(resultC[0]['color'], float(resultC[0]['prob']))
                    textM = "{}: {:.4f}".format(resultM[0]['brand'], float(resultM[0]['prob']))
                    cv2.putText(image, textC, (x + 2, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(image, textM, (x + 4, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                textL = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, textL, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        st.image(image, caption='Processed Image.', channels='BGR')
        end = time.time()
        st.write("Time took {:.6f} seconds".format(end - start))
# External files to download.
EXTERNAL_DEPENDENCIES = {
    "yolov4.weights": {
        "url": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights",
        "size": 257717640
    },
    "yolov4.cfg": {
        "url": "https://raw.githubusercontent.com/anggerwicaksono/car-color-classifier-yolo4-python/master/yolov4/yolov4.cfg",
        "size": 13351
    }
}

if __name__ == "__main__":
    main()
