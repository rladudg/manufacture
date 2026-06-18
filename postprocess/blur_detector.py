"""블러 감지 — OpenCV 후처리 알고리즘 2."""
import cv2

BLUR_THRESHOLD = 100.0   # 낮을수록 블러. 카메라 환경에 맞게 튜닝

def detect_blur(image, bbox=None):
    roi = image
    if bbox:
        x, y, w, h = bbox
        roi = image[y:y+h, x:x+w]
    if roi.size == 0:
        return 0.0, True

    gray     = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lap      = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(lap.var())
    return variance, variance < BLUR_THRESHOLD