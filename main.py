import cv2

from ultralytics import YOLO
model = YOLO("yolov8n.pt")

input_video_path = "testing.mp4"


output_video_path = "analyzed_video.mp4"

cap = cv2.VideoCapture(input_video_path)# for recorded video
#cap = cv2.VideoCapture(input_video_path)# for live video



class_list = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)

    person = 0
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            currentClass = class_list[int(box.cls[0])]
            currentConf = box.conf[0]

            if currentClass == 'person' and currentConf > 0.3:
                person += 1
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                cv2.putText(img, currentClass, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA, False)

    cv2.putText(img, "person: " + str(person), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA, False)
   
    cv2.imshow("Analyzed Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
