import cv2
print("Testing camera 0...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("FAILED to open camera 0")
else:
    ret, frame = cap.read()
    if ret:
        print(f"SUCCESS! Frame shape: {frame.shape}")
    else:
        print("Opened camera 0, but failed to read frame.")
cap.release()
