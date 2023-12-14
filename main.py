import os
from tqdm import tqdm
import cv2
import random
import json
from ultralytics import YOLO

from tracker import Tracker
from motmetrics import mm
import numpy as np
video_path = 'F:/USB/MSC_23_10_10/pia/object-tracking-yolov8-deep-sort-master/test.mov'
video_out_path = 'F:/USB/MSC_23_10_10/pia/object-tracking-yolov8-deep-sort-master/test_out.mp4'
json_path = 'F:/USB/MSC_23_10_10/pia/object-tracking-yolov8-deep-sort-master/test.json'

def get_data_from_json(json_path): 
    with open(json_path) as f: 
        data = json.load(f)
        objects = []
        for i in tqdm(range(len(data))):
            name = data[i]["name"];
            labels = data[i]["labels"];
            print("\n"+str(i)+".frame")
            object=[]
            for j in range(len(labels)):    
                obj = labels[j];
                if("box2d" in obj.keys()):
                    id = obj["id"];
                    id = id.lstrip("0")
                    category = obj["category"];
                    x1 = int(obj["box2d"]["x1"]);
                    y1 = int(obj["box2d"]["y1"]);
                    x2 = int(obj["box2d"]["x2"]);
                    y2 = int(obj["box2d"]["y2"]);
                    print(id,category, x1, y1, x2, y2)
                    object.append([id,category, x1, y1, x2, y2])
            objects.append(object)
    return objects   



cap = cv2.VideoCapture(video_path)



ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

model = YOLO("yolov8n.pt")

tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

detection_threshold = 0.5
predictions=[]
while ret:

    results = model(frame)

    for result in results:
        detections = []
        class_id = 0    
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score])
        
        tracker.update(frame, detections)
        trackers=[]
        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id
            trackers.append([track_id,class_id,x1, y1, x2, y2])
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
    predictions.append(trackers)
    cap_out.write(frame)
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()
