import cv2
cap = cv2.VideoCapture('output/annotated_traffic.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 60) # take 60th frame
ret, frame = cap.read()
if ret:
    cv2.imwrite('../rajesh-portfolio/public/images/smart_traffic.png', frame)
cap.release()
