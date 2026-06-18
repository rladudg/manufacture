# Manufacturing Defect Inspection — Vision & MLOps Module

컨베이어 기반 제조 검사 시스템의 **비전/ML 파이프라인**.
YOLOv8 + OpenCV 하이브리드 2단계 판정으로 전자부품을 **PASS / DEFECT / HOLD** 로 분류한다.

> 졸업 캡스톤 프로젝트(5인 팀, *Data-Mogi*) 중 **ML Engineer(비전/MLOps)** 담당 영역.
> 전체 파이프라인: Arduino(제어) → Raspberry Pi(엣지·영상) → C# WinForms(통합 허브) → **본 비전 서비스** → MySQL / Flask 대시보드

---

## 내가 구현한 것 (My Contribution)

이 저장소는 전체 시스템에서 **비전 검사 엔진과 MLOps 루프**를 담당한다. 구체적으로:

- **YOLOv8 학습 파이프라인** — 9-class 스킴(부품 body + 부품별 핀 품질 클래스) 설계, Roboflow 기반 데이터셋 관리, Ultralytics 학습/검증
- **YOLO + OpenCV 하이브리드 2단계 판정 로직** — YOLO를 "후보 탐지(proposal)"로 제한하고 OpenCV가 "확정(confirmation)"하는 구조 설계 및 구현
- **OpenCV 후처리 알고리즘** — 핀 개수 검출, Laplacian variance 기반 blur 감지, ROI 정렬 검증
- **HOLD(불확실성) 처리** — 이중 confidence 임계값 기반 분기 로직
- **HTTP inference service** — C# 통합 허브가 호출하는 추론 API(Flask, port 5002)
- **Active learning 루프** — 운영 중 오분류 자동 수집 → 재라벨링 → 재학습 파이프라인

---

## 핵심 설계

### 2단계 판정 (proposal → confirmation)
YOLO 단독으로 세부 결함까지 판정하지 않는다.

1. **proposal** — YOLOv8s가 부품 위치/종류 후보 탐지
2. **confirmation** — OpenCV 후처리가 핀 개수·blur·ROI로 최종 확정

산업 비전 솔루션(Cognex VisionPro 등)의 *detector + classifier* 구조와 동일한 철학이다. 모던 ML과 전통 CV를 결합해 YOLO의 약점(미세 결함, 핀 개수 같은 정밀 판정)을 규칙 기반으로 보완한다.

### HOLD = 불확실성 분리
- HOLD는 YOLO 클래스 라벨이 **아니라** confidence 임계 기반 출력 카테고리
- 이중 임계값: `CONFIRM_THRESHOLD`(예: 0.75) / `HOLD_THRESHOLD`(예: 0.50)
- 애매한 품목을 강제로 PASS/DEFECT 하지 않고 HOLD로 분리 → 오분류 비용(특히 불량 누락)을 최소화

---

## 저장소 구조

---

## 모델 & 데이터셋

| 항목 | 내용 |
|---|---|
| 모델 | YOLOv8s (9-class) |
| 데이터셋 | 고정 환경 촬영, Roboflow 관리 |
| Augmentation | brightness / exposure / blur + rotation ±5° **한정** |
| 미사용 | Mosaic / Mixup / flip (고정 환경 원칙 유지를 위해 의도적 배제) |
| 목표 성능 | mAP@50 ~0.85 |
| 배포 | ONNX export (opset=12, C# OnnxRuntime 호환) |

> **모델 선택 근거**: 약 500~1,000장 규모 데이터셋에는 YOLOv8s가 적합. YOLOv8n은 엣지/모바일 타겟용이라 정확도가 부족하다.

---

## 서비스 연동

- Flask HTTP inference service (port 5002)
- 엔드포인트: `/health`, `/infer/cam1`
- C# WinForms 통합 허브가 검사 시점에 호출, **500ms 타임아웃 시 HOLD로 처리**(fail-safe)

---

## MLOps

- 데이터셋 버전 관리 (v1 → v2 → v3)
- Active learning: 운영 중 오분류 자동 수집 → 주기적 재라벨링 → 재학습 후 성능 비교
- Confusion matrix + failure case 분석 기반 개선 사이클

---

## Tech Stack

`Python` · `Ultralytics YOLOv8` · `OpenCV` · `Roboflow` · `Flask` · `ONNX Runtime`

---

## 전체 시스템 맥락

본 저장소는 5인 팀이 ISA-95 Purdue Model을 참조해 설계한 제조 검사 시스템의 일부.
비전 모듈 외 구성요소(Arduino 펌웨어, RPi 엣지, C# HMI, MQTT broker, Flask 대시보드)는 별도 영역에서 다룸.
