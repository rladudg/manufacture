"""통합 판정 로직 — C#팀원이 이 파일 기준으로 포팅."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from pin_counter   import count_pins
from blur_detector import detect_blur
from roi_validator import is_roi_valid

HOLD_CONFIDENCE_THRESHOLD = 0.70
PASS_CONFIDENCE_THRESHOLD = 0.85

def decide_verdict(image, yolo_result):
    metrics = {
        'yolo_class':          yolo_result.get('top_class'),
        'yolo_confidence':     yolo_result.get('top_confidence'),
        'opencv_pin_count':    None,
        'opencv_blur_score':   None,
        'opencv_roi_centered': None,
    }

    top_class      = yolo_result.get('top_class')
    top_confidence = yolo_result.get('top_confidence', 0.0)
    detections     = yolo_result.get('detections', [])

    # 1. HOLD — 신뢰도 낮음
    if top_confidence < HOLD_CONFIDENCE_THRESHOLD:
        return 'hold', f'Low confidence: {top_confidence:.2f}', metrics

    # 2. HOLD — hold_candidate
    if top_class == 'hold_candidate':
        return 'hold', 'hold_candidate by YOLO', metrics

    # 3. HOLD — 탐지 없음
    if not detections:
        return 'hold', 'No detections', metrics

    # 4. OpenCV 분석
    b    = detections[0]['bbox']
    bbox = (b['x'], b['y'], b['w'], b['h'])

    blur_score, is_blurry = detect_blur(image, bbox)
    metrics['opencv_blur_score'] = blur_score
    if is_blurry:
        return 'hold', f'Blur detected (score={blur_score:.1f})', metrics

    centered, partial = is_roi_valid(image.shape, bbox)
    metrics['opencv_roi_centered'] = centered
    if partial:
        return 'hold', 'Partial ROI', metrics

    # 5. 결함 확정
    has_pin_defect = any(d['class']=='pin_defect_candidate'
                         and d['confidence']>0.6 for d in detections)
    has_surface    = any(d['class']=='surface_defect_candidate'
                         and d['confidence']>0.6 for d in detections)

    if has_pin_defect:
        expected  = 3 if top_class=='transistor' else 2
        pin_count, _ = count_pins(image, bbox, top_class)
        metrics['opencv_pin_count'] = pin_count
        if pin_count != expected:
            return 'defect', f'Pin {pin_count} != {expected}', metrics
        return 'hold', f'YOLO suspects but OpenCV OK ({pin_count})', metrics

    if has_surface:
        return 'defect', 'Surface defect confirmed', metrics

    # 6. 정상
    if top_class in ('transistor','capacitor','regulator') \
       and top_confidence >= PASS_CONFIDENCE_THRESHOLD:
        expected  = 3 if top_class=='transistor' else 2
        pin_count, _ = count_pins(image, bbox, top_class)
        metrics['opencv_pin_count'] = pin_count
        if pin_count == expected:
            return 'pass', 'All checks passed', metrics
        return 'defect', f'Pin mismatch: {pin_count} vs {expected}', metrics

    return 'hold', 'Fallback hold', metrics

if __name__ == '__main__':
    import cv2
    import numpy as np

    img = np.zeros((480, 640, 3), dtype=np.uint8)

    yolo_fake = {
        'top_class':      'transistor',
        'top_confidence': 0.92,
        'detections': [{
            'class':      'transistor',
            'confidence': 0.92,
            'bbox': {'x': 160, 'y': 120, 'w': 320, 'h': 240}
        }]
    }

    verdict, detail, metrics = decide_verdict(img, yolo_fake)
    print(f"Verdict : {verdict}")
    print(f"Detail  : {detail}")
    print(f"Metrics : {metrics}")