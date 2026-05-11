# QuickDraw Stroke-by-Stroke 실시간 인식 시스템

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hyeseonghwang0017-coder/quickdraw_stroke_by_stroke_classification/blob/main/classification_final.ipynb)

Google Quick, Draw! 데이터셋으로 **그림을 그리는 도중 획(stroke)이 추가될 때마다** 실시간으로 클래스를 예측하는 시스템입니다.

일반 CNN은 완성된 이미지를 받아 예측하지만, 실시간 시나리오에서는 초기 획만 있는 상태에서도 높은 확신으로 잘못된 예측을 내놓습니다. 이를 해결하기 위해 세 가지 기준을 모두 충족할 때만 예측을 확정합니다:

1. **최소 완성도 비율** — 클래스별 평균 획수 기반으로 자동 계산 (통계 초기 추정 → Val 데이터 미세조정)
2. **최소 획수** — 클래스별 train 데이터 10th percentile 기반
3. **클래스별 최적 신뢰도 임계값** — Val 데이터에서 `[0.60, 0.65, ..., 0.95]` 그리드 탐색 후 정확도 최고 값 선택

---

## 모델 및 데이터

**모델**: MobileNetV2 (ImageNet pre-trained) + Dropout(0.3) + Linear  
**파라미터**: 2,236,682개

**데이터**: 탈것 10개 클래스, 클래스당 10,000개 (총 100,000개)

| 범주 | 클래스 |
|------|--------|
| 하늘 | airplane, helicopter, hot air balloon |
| 물 | sailboat, submarine, canoe |
| 땅 | car, bus, train, motorbike |

**분할**: Train 70% (70,000) / Val 15% (15,000) / Test 15% (15,000)

---

## 학습 설정

```python
BATCH_SIZE = 128
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.3
IMG_SIZE = 256
TOP_K = 3
CONFIDENCE_CANDIDATES = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
```

- Optimizer: Adam + ReduceLROnPlateau (factor=0.5, patience=2)
- Loss: CrossEntropyLoss
- Train augmentation: RandomRotation(15°), RandomAffine, RandomErasing

---

## 결과

**최고 Val Accuracy**: 96.11%  
**Test Accuracy (Stroke-by-Stroke)**: 91.29%

| Class | Accuracy |
|-------|----------|
| hot air balloon | 96.4% |
| motorbike | 96.0% |
| airplane | 95.5% |
| submarine | 93.8% |
| sailboat | 92.4% |
| bus | 92.4% |
| helicopter | 90.9% |
| car | 86.6% |
| canoe | 86.1% |
| train | 83.0% |
| **OVERALL** | **91.29%** |

---

## 사용법 (Google Colab)

셀을 순서대로 실행합니다. 학습된 모델과 임계값은 **본인의 Google Drive**에 저장되어 재실행 시 자동으로 불러옵니다.

> **처음 실행하는 경우**: `best_model.pth`와 `thresholds.json`이 없으므로 셀 9에서 학습이 시작됩니다 (Tesla T4 기준 약 30~40분 소요). 완료 후에는 Drive에 저장되어 이후 실행 시 자동으로 스킵됩니다.

```
셀 1-2   : 라이브러리 및 하이퍼파라미터
셀 3     : 데이터 다운로드
셀 4-6   : 전처리 및 DataLoader
셀 7-9   : 모델 정의 및 학습 (Drive에 best_model.pth 저장)
셀 11    : 클래스별 최소 완성도 및 최소 획수 계산
셀 13-14 : 클래스별 최적 신뢰도 계산 (thresholds.json 저장)
셀 15    : Top-K 예측 데모
셀 16-18 : Stroke-by-Stroke 평가 및 결과 출력
```

---

## 파일 구조

```
quickdraw_stroke_by_stroke_classification/
├── classification_final.ipynb   # 전체 파이프라인
└── README.md
```

Google Drive에 자동 저장:
- `best_model.pth` — 최고 Val Acc 모델 가중치
- `history.json` — 학습 곡선
- `thresholds.json` — MIN_COMPLETION_RATIOS, OPTIMAL_CONFIDENCES

---

## 데이터 출처

[Google Quick, Draw! Dataset]: https://github.com/googlecreativelab/quickdraw-dataset
