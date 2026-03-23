# YOLO model 연동 (You Only Look Once)
# 실시간 객체 탐지를 위해 설계된 딥러닝 모델, ONYX 파일로 실행
import cv2

net = cv2.dnn.readNetFromONNX("yolov5s.onnx")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()

    # preds shape: (1, 25200, 85)
    # 25200개의 바운딩박스 후보 중 confidence 높은 것만 필터링
    import numpy as np
    outputs = preds[0]  # (25200, 85)

    boxes, confidences, class_ids = [], [], []
    h, w = frame.shape[:2]

    for det in outputs:
        confidence = det[4]  # 객체일 확률
        if confidence < 0.5:
            continue

        scores = det[5:]           # 80개 클래스별 확률
        class_id = np.argmax(scores)  # 가장 높은 클래스
        if scores[class_id] < 0.5:
            continue

        # YOLOv5 출력은 640x640 기준 중심좌표(cx, cy, w, h) 형태
        cx, cy, bw, bh = det[0], det[1], det[2], det[3]
        x1 = int((cx - bw / 2) * w / 640)
        y1 = int((cy - bh / 2) * h / 640)
        x2 = int((cx + bw / 2) * w / 640)
        y2 = int((cy + bh / 2) * h / 640)

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        confidences.append(float(confidence))
        class_ids.append(class_id)

    # NMS(Non-Maximum Suppression): 겹치는 박스 중 가장 좋은 것만 남김
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    CLASSES = open("/Users/yoo/Documents/project_openCV/4chapter/coco.names").read().strip().split("\n") \
        if __import__("os").path.exists("/Users/yoo/Documents/project_openCV/4chapter/coco.names") \
        else [str(i) for i in range(80)]

    for i in indices:
        x, y, bw, bh = boxes[i]
        label = f"{CLASSES[class_ids[i]]}: {confidences[i]*100:.1f}%"
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLO Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
