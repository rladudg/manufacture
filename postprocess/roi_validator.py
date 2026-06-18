"""ROI 위치 검증 — OpenCV 후처리 알고리즘 3."""

def is_roi_valid(image_shape, bbox, edge_margin=20):
    h, w    = image_shape[:2]
    x, y, bw, bh = bbox

    partial  = (x < edge_margin or y < edge_margin or
                (x+bw) > (w-edge_margin) or (y+bh) > (h-edge_margin))

    cx       = x + bw/2
    cy       = y + bh/2
    centered = (abs(cx - w/2) < w*0.3 and abs(cy - h/2) < h*0.3)

    return centered, partial