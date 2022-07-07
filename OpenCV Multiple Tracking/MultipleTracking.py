from os.path import isfile, join
import os
import cv2
import numpy as np
import pandas as pd

# pathIn = r"./MOT17-04-SDP/img1"
# pathOut = "./MOT17-04-SDP/MOT17-04-SDP.mp4"
#
# files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
#
# fps = 30
# size = (1920, 1080)
# out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc("m", "p", "4", "v"), fps, size, True)
#
# for i in files:
#
#     filename = pathIn + "\\" + i
#
#     img = cv2.imread(filename)
#     out.write(img)
#
# out.release()

OPENCV_OBJECT_TRACKERS = {"csrt": cv2.TrackerCSRT.create,
                          "kcf": cv2.TrackerKCF.create,
                          "boosting": cv2.TrackerBoosting.create,
                          "mil": cv2.TrackerMIL.create,
                          "tld": cv2.TrackerTLD.create,
                          "medianflow": cv2.TrackerMedianFlow.create,
                          "mosse": cv2.TrackerMOSSE.create}

video_path = "C:\\Users\\Halil Umut Yalçın\\Desktop\\ \\Artificial Intelligence\\OpenCv\\OpenCV ile Nesne Takibi\\MOT17-04-SDP\\MOT17-04-SDP.mp4"

tracker_name = "mil"

trackers = cv2.MultiTracker_create()

cap = cv2.VideoCapture(video_path)

fps = 30
f = 0

while 1:
    ret, frame = cap.read()

    (H, W) = frame.shape[:2]

    frame = cv2.resize(frame, (960, 540))

    success, boxes = trackers.update(frame)

    info = [("Tracker", tracker_name),
            ("Success", "Yes" if success else "No")]

    string_text = ""

    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        string_text = string_text + text + ""

    cv2.putText(frame, string_text, (10, H - (i * 20) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("t"):
        box = cv2.selectROI("Frame", frame, fromCenter=False)

        tracker = OPENCV_OBJECT_TRACKERS[tracker_name]()
        trackers.add(tracker, frame, box)
    elif key == ord("q"): break

    f += 1

cap.release()
cv2.destroyAllWindows()