# KCC 2025: Feature Similarity Contrastive Learning 기반 불균형 Fine-Grained 야생동물 이미지 분류

**저자: 이서희, 박가람, 손규원, 도희찬, 양희성, 박혜영**
**소속: 경북대학교 전자공학부, 경북대학교 컴퓨터학부**

본 저장소는 2025 한국컴퓨터종합학술대회(KCC 2025)에서 발표된 위 논문의 요약 자료입니다.

## 1. Abstract (요약)

> [cite_start]야생동물 영상의 종 분류는 생태계 보전 정책 수립에 필수적이지만, **데이터 불균형(Imbalance)**과 **Fine-grained 클래스 간의 높은 시각적 유사성**(e.g., 고라니 vs 노루)으로 인해 높은 정확도를 달성하기 어렵습니다[cite: 13, 21, 22]. [cite_start]본 연구에서는 이 두 가지 문제를 동시에 해결하기 위해, **중간 계층(intermediate layer) feature의 유사도에 기반한 Contrastive Loss**를 기존 분류 손실 함수(Cross-Entropy)에 결합하는 새로운 학습 방법을 제안합니다[cite: 14, 15, 33]. [cite_start]이 방법은 Oversampling과 같은 데이터 샘플링 기법과 쉽게 결합될 수 있으며, 별도의 fine-tuning 없이 **End-to-End 학습**이 가능합니다[cite: 17, 34].

## 2. Problem Statement (문제 정의)

1.  [cite_start]**데이터 불균형 (Data Imbalance):** 자연 환경에서 수집된 데이터는 종별 출현 빈도가 심각하게 불균형하여, 소수 클래스(minority class)에 대한 분류 성능이 저하됩니다[cite: 21].
2.  [cite_start]**Fine-Grained 분류:** 고라니(Water deer)와 노루(Roe deer)처럼 외형적으로 매우 유사한 종들을 구분하는 것은 분류 모델에게 매우 어려운 작업입니다[cite: 22, 23].

[cite_start]기존의 Oversampling 기법은 fine-grained 문제 해결에 한계가 있으며 [cite: 94][cite_start], 기존 Contrastive Learning 방식은 별도의 fine-tuning 단계가 필요해 End-to-End 학습이 어렵다는 단점이 있습니다[cite: 31].

## 3. Proposed Method (제안하는 방법)

본 연구는 두 문제를 동시에 완화하기 위해 **Feature Similarity Contrastive Learning**을 제안합니다.

[cite_start]핵심 아이디어는 ResNet-18과 같은 Backbone 네트워크의 **중간 계층(e.g., Layer 3, 4)에서 추출된 Feature**를 직접 활용하는 것입니다[cite: 37, 38, 58].

1.  [cite_start]배치 내의 이미지들이 네트워크를 통과할 때, 중간 계층의 Feature map을 **Global Average Pooling(GAP)**하여 feature vector $h$를 추출합니다[cite: 41, 42, 59].
2.  [cite_start]배치 내 모든 이미지 쌍($i, j$) 간의 코사인 유사도를 계산하여 **Feature Similarity Matrix ($S^l$)**를 만듭니다[cite: 60, 61].
3.  동시에, 실제 레이블 $c$를 기반으로 **Target Similarity Matrix ($T$)**를 생성합니다. (i와 j의 클래스가 같으면 $T_{ij}=1$, 다르면 $T_{ij}=0$) [cite_start][cite: 63, 64].
4.  [cite_start]**Contrastive Loss ($\mathcal{L}_c^l$)**는 이 두 행렬($S^l$와 $T$) 간의 **MSE(Mean Squared Error)**로 정의됩니다[cite: 72].
5.  [cite_start]최종 손실 함수 $\mathcal{L}_{total}$는 기존의 분류 손실 함수(CrossEntropy)와 모든 중간 계층의 Contrastive Loss를 합산하여 구성됩니다[cite: 49, 72].

$$
\mathcal{L}_{total} = \text{CrossEntropy}(y, y^*) + \sum_{l} \mathcal{L}_{c}^{l} \quad \text{where} \quad \mathcal{L}_{c}^{l} = \text{MSE}(S^l, T)
$$

[cite_start]이 과정은 [그림 2]와 같이 동일 클래스(e.g., Water deer)의 feature들은 서로 가깝게 당기고(Pull), 다른 클래스(e.g., Roe deer)의 feature들은 멀어지도록(Push) 유도하여 [cite: 67, 68, 69, 74][cite_start], fine-grained 클래스 간의 차별적인 특징 학습을 촉진합니다[cite: 75].

![제안하는 방법의 구조](images/figure1.png)
> [cite_start]그림 1: 제안하는 학습 방법의 구조 [cite: 53]

## 4. Experiments & Results (실험 결과)

[cite_start]제안하는 방법의 성능 검증을 위해 야생동물 데이터셋 [cite: 81]을 두 가지 조건으로 실험했습니다.
* [cite_start]**Coarse-grained (Imbalanced):** 고라니 vs 멧돼지 (1:1 ~ 1:0.001 비율) [cite: 82]
* [cite_start]**Fine-grained (Imbalanced):** 고라니 vs 노루 (1:1 ~ 1:0.001 비율) [cite: 83]

[cite_start][그림 3]과 [표 1]에서 볼 수 있듯이, Baseline(CE Loss)이나 Oversampling만 적용한 방법은 데이터 불균형이 심화될수록(특히 fine-grained 조건에서) 정확도가 급격히 하락했습니다[cite: 111, 112].

[cite_start]반면, **Oversampling과 제안하는 Contrastive Loss를 함께 사용한 모델**은 1:0.005 (10000개 vs 50개)의 극심한 불균형 조건에서도 높은 분류 정확도를 안정적으로 유지했습니다[cite: 95, 113, 114]. [cite_start]이는 제안하는 방법이 단순한 데이터 분포 조정(Oversampling)만으로는 얻기 힘든 효과적인 feature 분리를 달성했음을 입증합니다[cite: 96, 119].

![실험 결과 그래프](images/figure3.png)
> [cite_start]그림 3: 데이터 불균형 비율에 따른 클래스별 정확도 [cite: 104]

![Ablation Study 표](images/table1.png)
> [cite_start]표 1: 불균형 데이터 조건에서 방법 별 클래스 정확도 (Ablation Study) [cite: 106]

## 5. Conclusion (결론)

[cite_start]본 연구는 데이터 불균형과 fine-grained 분류 문제를 동시에 해결하기 위해, oversampling과 중간 계층 feature 기반 contrastive loss를 결합한 학습 방법을 제안했습니다[cite: 116, 117]. 제안 방법은 별도의 fine-tuning 없이 End-to-End 학습이 가능하며, 다양한 불균형 조건에서 기존 방법들보다 우수한 성능을 보였습니다. [cite_start]이는 향후 생태계 모니터링 및 다양한 fine-grained 희귀 클래스 인식 분야에 기여할 수 있을 것으로 기대됩니다[cite: 18, 120].
