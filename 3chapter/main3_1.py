import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# 객체 추적 Tracking
# 한 번 인식된 객체를 프레임 간에 계속 추적함.
bbox = cv2.selectROI("Tracking", frame, False) # 추적할 객체를 직접 지정, (x,y,w,h) 반환
if bbox == (0,0,0,0):
    cap.release()
    cv2.destroyAllWindows()
    exit()
tracker = cv2.TrackerKCF.create()   # 추적기, KCF:픽셀 패턴 학습 후 추적, 빠르고 가벼움
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()
    success, box = tracker.update(frame)

    if success:
        x, y, w, h = map(int, box)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Tracking", (x, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Lost", (x, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
