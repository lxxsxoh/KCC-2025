# KCC-2025

# KCC 2025: Feature Similarity Contrastive Learning 기반 불균형 Fine-Grained 야생동물 이미지 분류

[cite_start][KCC 2025](https://www.kiise.or.kr/conf/kcc/2025/)에서 발표된 "Feature Similarity Contrastive Learning 기반 불균형 Fine-Grained 야생동물 이미지 분류 방법" [cite: 48] 논문의 공식 Pytorch 구현입니다.

## 1. Abstract

[cite_start]야생동물 영상 분류는 생태계 보전 정책에 필수적이지만 [cite: 58, 64][cite_start], **데이터 불균형(Imbalance)** 문제와 [cite: 66] [cite_start]**Fine-grained 클래스 간의 높은 유사성** (e.g., 고라니 vs 노루) [cite: 58, 67, 68] 때문에 정확도 확보가 어렵습니다.

[cite_start]본 연구는 이 두 가지 문제를 동시에 해결하기 위해, 표준적인 Cross-Entropy 손실 함수에 **중간 계층(Intermediate layer)의 Feature 유사도에 기반한 Contrastive Loss**를 결합하는 새로운 학습 방법을 제안합니다[cite: 59, 60, 78].

[cite_start]이 방식은 Oversampling 기법과 쉽게 결합할 수 있으며 [cite: 60, 79][cite_start], 별도 fine-tuning 없이 **End-to-End 학습**이 가능합니다[cite: 62, 79]. [cite_start]실험 결과, 제안하는 방법은 Coarse-grained 및 Fine-grained 불균형 조건 모두에서 기존 방식 대비 높은 분류 정확도를 달성했습니다[cite: 61, 163].

## 2. Proposed Method

[cite_start]제안하는 방법의 구조는 [그림 1]과 같습니다[cite: 98]. [cite_start]Backbone 네트워크(ResNet-18) [cite: 129][cite_start]의 중간 계층(e.g., Layer 3, 4)에서 Feature map을 추출합니다[cite: 103].
[cite_start]각 Feature map을 GAP(Global Average Pooling)하여 feature vector를 얻고 [cite: 104][cite_start], 배치 내 모든 이미지 쌍의 코사인 유사도를 계산하여 **Feature Similarity Matrix ($S^l$)**를 만듭니다[cite: 105, 106].
[cite_start]동시에, 레이블을 기반으로 **Target Similarity Matrix ($T$)**를 생성합니다[cite: 108]. ($T_{ij}$는 $c_i == c_j$일 때 1, 아닐 때 0) [cite_start][cite: 109]

[cite_start]최종 손실 함수 $\mathcal{L}_{total}$는 이 두 행렬 간의 **MSE Loss (Contrastive Loss)**와 원본 **Cross-Entropy Loss**를 결합하여 계산됩니다[cite: 117].

$$
\mathcal{L}_{total} = \text{CrossEntropy}(y, y^*) + \sum_{l} \mathcal{L}_{c}^{l} \quad \text{where} \quad \mathcal{L}_{c}^{l} = \text{MSE}(S^l, T)
$$

[cite_start]이 과정은 같은 클래스의 feature는 가깝게(Pull), 다른 클래스의 feature는 멀게(Push) 만들어 [cite: 112, 113, 114] [cite_start]fine-grained 클래스 간의 차별적 특징 학습을 촉진합니다[cite: 119, 120].

> [cite_start]그림 1: 제안하는 학습 방법의 구조 [cite: 98]
