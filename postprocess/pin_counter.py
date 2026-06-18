"""핀 벌어짐 분석 — OpenCV 후처리 (cam_test.py analyze_pins 통합)"""
import cv2
import numpy as np


def analyze_pins(frame, body_box):
    """
    body bbox 영역을 크롭해서 OpenCV로 핀 벌어짐 판단.
    반환: 'defective' | 'normal' | None (판단 불가)
    """
    x1, y1, x2, y2 = map(int, body_box)

    h_img, w_img = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 길쭉한 윤곽선(핀 후보) 필터링
    pin_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / (w + 1e-5)
        area = cv2.contourArea(cnt)
        if aspect_ratio > 1.8 and area > 8:
            pin_contours.append(cnt)

    if len(pin_contours) < 2:
        return None  # 핀을 충분히 못 찾음 → 판단 불가

    # 핀 중심 X 좌표 수집
    centers = []
    for cnt in pin_contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            centers.append(cx)

    if len(centers) < 2:
        return None

    centers.sort()
    max_gap   = max(centers[i+1] - centers[i] for i in range(len(centers) - 1))
    roi_width = x2 - x1
    threshold = roi_width * 0.3

    return "defective" if max_gap > threshold else "normal"


def count_pins(image, bbox, product_type='transistor'):
    """기존 핀 카운트 함수 (algorithm_reference 에서 사용)"""
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]
    if roi.size == 0:
        return 0, 0.0

    pin_roi = roi[int(h*0.7):, :]
    gray = cv2.cvtColor(pin_roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    max_area   = pin_roi.shape[0] * pin_roi.shape[1] * 0.3
    valid_pins = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 20 <= area <= max_area:
            _, _, cw, ch = cv2.boundingRect(cnt)
            if ch / max(cw, 1) >= 0.8:
                valid_pins.append(cnt)

    pin_count  = len(valid_pins)
    expected   = 3 if product_type == 'transistor' else 2
    confidence = max(0.0, 1.0 - abs(pin_count - expected) * 0.3)
    return pin_count, confidence