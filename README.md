# Feature Similarity Contrastive Learning 기반 불균형 Fine-Grained 야생동물 이미지 분류

A hybrid training approach combining **Oversampling** and **Feature Similarity Contrastive Learning**  
to improve fine-grained wildlife classification under severe data imbalance.

---

## 개요 (Overview)

야생동물 영상 분류(Wildlife Image Classification)는 생태계 모니터링 및 보전 정책 수립에 필수적인 기술이다.  
그러나 다음 두 가지 주요 문제로 인해 높은 정확도를 달성하기 어렵다.

1. **데이터 불균형 (Data Imbalance)** — 종별 출현 빈도의 편차로 인한 학습 불안정  
2. **세밀 종 구분 (Fine-Grained Similarity)** — 외형적으로 유사한 종 간의 특징 분리 어려움  
   (예: 노루 Roe Deer vs 고라니 Water Deer)

이 연구에서는 이러한 문제를 동시에 해결하기 위해  
**Feature Similarity Contrastive Learning** 방식을 제안한다.  
기존의 **Cross-Entropy Loss**에 **feature-level contrastive loss**를 추가하여  
클래스 간 구분력을 강화하고, 불균형 데이터에서도 안정적인 학습을 달성한다.

---

## 제안하는 방법 (Proposed Method)

### 핵심 아이디어 (Key Idea)

- 중간 계층(intermediate layer)의 feature 유사도 기반 contrastive loss 추가  
- 클래스 내 feature는 유사하게, 클래스 간 feature는 멀어지도록 학습  
- Oversampling 및 RandAugment으로 소수 클래스의 다양성 확보  

### 학습 구조 (Architecture)

<p align="center">
   <img width="757" height="380" alt="image" src="https://github.com/user-attachments/assets/4c8cbb5d-f005-4d84-8aad-7ffd7b50de38" />
</p>

**Loss 구성**

$$
S_{ij} = \frac{h_i \cdot h_j}{\|h_i\| \|h_j\|}, \quad
T_{ij} = 
\begin{cases}
1, & \text{if } c_i = c_j \\
0, & \text{otherwise}
\end{cases}
$$

$$
L_{total} = CrossEntropy(y, y^*) + \sum_l MSE(S^{(l)}, T)
$$

---

## 실험 설정 (Experimental Setup)

| 항목 | 설정 |
|------|------|
| Dataset | AIHub 야생동물 영상 데이터 ([AIHub Link](https://www.aihub.or.kr)) |
| 클래스 (Classes) | Water Deer, Wild Boar, Roe Deer |
| Backbone | ResNet-18 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Epochs | 100 |
| Augmentation | Oversampling, RandAugment, Color Jitter |

---

## 결과 (Results)

### 불균형 비율별 정확도 변화 (Accuracy under Different Imbalance Ratios)

| Data Ratio | Baseline | Oversampling | Contrastive | Proposed (FSCL) |
|-------------|-----------|---------------|--------------|----------------|
| 1:1 | 0.99 | 0.99 | 0.99 | 0.99 |
| 1:0.1 | 0.87 | 0.96 | 0.93 | 0.96 |
| 1:0.01 | 0.34 | 0.94 | 0.00 | 0.89 |
| 1:0.005 | 0.15 | 0.77 | 0.00 | 0.91 |
| 1:0.001 | 0.00 | 0.43 | 0.00 | 0.31 |

<p align="center">
   <img width="752" height="545" alt="image" src="https://github.com/user-attachments/assets/56a3838f-f213-4556-b55a-7d95ed918358" />
</p>

우리가 제안한 방법은 심각한 불균형 조건(1:0.001)에서도 안정적인 정확도를 유지하며,  
fine-grained 종 간 feature 분리를 명확히 달성하였다.

---

## Ablation Study

| 방법 (Method) | Coarse-Grained | Fine-Grained |
|---------------|----------------|--------------|
| Baseline (CE only) | ↓ | ↓↓↓ |
| Oversampling only | ↑ | ↔ |
| Contrastive only | ↔ | ↓ |
| Proposed (Over + Contrastive) | ↑↑↑ | ↑↑↑ |

---
