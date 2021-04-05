import cv2
import numpy as np
from tracker import Tracker

net = cv2.dnn.readNet('models/vehicle_0202/vehicle-detection-0202.bin',
                      'models/vehicle_0202/vehicle-detection-0202.xml')
# net.setPreferableBackend(cv2.DNN_BACKEND_INFERENCE_ENGINE)
my_tracker = Tracker()
cap = cv2.VideoCapture('rtsp://admin:parol12345@192.168.4.220:554/cam/realmonitor?channel=1&subtype=1')
# cv2.namedWindow('frame', cv2.WINDOW_GUI_EXPANDED)
cars = []
counter = 0


def calc_centroid(x1, y1, x2, y2):
    return (x2 + x1) // 2, (y2 + y1) // 2


def calc_distance(c1, c2):
    return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


track_list = []

while cv2.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    y_size = frame.shape[0]
    x_size = frame.shape[1]
    if not hasFrame:
        break
    st = cv2.getTickCount()

    blob = cv2.dnn.blobFromImage(frame, size=(672, 384))
    # blob = cv2.dnn.blobFromImage(frame, size=(400, 400))
    net.setInput(blob)
    out = net.forward()
    cords = []
    for detection in out.reshape(-1, 7):
        confidence = float(detection[2])
        xmin = int(detection[3] * frame.shape[1])
        ymin = int(detection[4] * frame.shape[0])
        xmax = int(detection[5] * frame.shape[1])
        ymax = int(detection[6] * frame.shape[0])

        x, y = calc_centroid(xmin, ymin, xmax, ymax)

        if confidence >= 0.9:
            cords.append((xmin, ymin, xmax, ymax))
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(255, 150, 150), thickness=5)
    vehicles, tails = my_tracker.update(cords)
    for ids, cords in vehicles.items():
        cv2.putText(frame, f'id: {ids}', (cords[0], cords[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 2)
        for tail in tails[ids]:
            cv2.circle(frame, tail, 2, (50, 100, 255), -2)
        cv2.circle(frame, calc_centroid(*cords), int(0.2 * (cords[3] - cords[1])), (0, 255, 0), 2)
        cv2.rectangle(frame, (cords[0], cords[1]), (cords[2], cords[3]), color=(150, 150, 150), thickness=2)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - st)
    cv2.putText(frame, 'FPS: {}'.format(str(round(fps, 2))), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    cv2.imshow('frame', frame)
