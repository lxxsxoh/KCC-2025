# KCC-2025

KCC 2025: Feature Similarity Contrastive Learning 기반 불균형 Fine-Grained 야생동물 이미지 분류KCC 2025에서 발표된 "Feature Similarity Contrastive Learning 기반 불균형 Fine-Grained 야생동물 이미지 분류 방법" 논문 7의 공식 Pytorch 구현입니다.1. Abstract야생동물 영상 분류는 생태계 보전 정책에 필수적이지만, 데이터 불균형(Imbalance) 문제와 Fine-grained 클래스 간의 높은 유사성 (e.g., 고라니 vs 노루) 때문에 정확도 확보가 어렵습니다8888.본 연구는 이 두 가지 문제를 동시에 해결하기 위해, 표준적인 Cross-Entropy 손실 함수에 중간 계층(Intermediate layer)의 Feature 유사도에 기반한 Contrastive Loss를 결합하는 새로운 학습 방법을 제안합니다999999999.이 방식은 Oversampling 기법과 쉽게 결합할 수 있으며 10, 별도 fine-tuning 없이 End-to-End 학습이 가능합니다11111111. 실험 결과, 제안하는 방법은 Coarse-grained 및 Fine-grained 불균형 조건 모두에서 기존 방식 대비 높은 분류 정확도를 달성했습니다12121212.2. Proposed Method제안하는 방법의 구조는 [그림 1]과 같습니다. Backbone 네트워크(ResNet-18) 13의 중간 계층(e.g., Layer 3, 4)에서 Feature map을 추출합니다14141414.각 Feature map을 GAP(Global Average Pooling) 15하여 feature vector를 얻고, 배치 내 모든 이미지 쌍의 코사인 유사도를 계산하여 **Feature Similarity Matrix ($S^l$)**를 만듭니다16.동시에, 레이블을 기반으로 **Target Similarity Matrix ($T$)**를 생성합니다17. ($T_{ij}$는 $c_i == c_j$일 때 1, 아닐 때 0)최종 손실 함수 $\mathcal{L}_{total}$는 이 두 행렬 간의 **MSE Loss (Contrastive Loss)**와 원본 Cross-Entropy Loss를 결합하여 계산됩니다18.$$\mathcal{L}_{total} = \text{CrossEntropy}(y, y^*) + \sum_{l} \mathcal{L}_{c}^{l} \quad \text{where} \quad \mathcal{L}_{c}^{l} = \text{MSE}(S^l, T)
$$[cite\_start]이 과정은 같은 클래스의 feature는 가깝게(Pull), 다른 클래스의 feature는 멀게(Push) 만들어 fine-grained 클래스 간의 차별적 특징 학습을 촉진합니다[cite: 101, 102, 103, 108, 109].

> [cite\_start]그림 1: 제안하는 학습 방법의 구조 [cite: 87]

## 3\. Installation

```bash
# 1. 저장소 클론
git clone https://github.com/lxxsxoh/KCC-2025.git
cd KCC-2025

# 2. (선택) 가상 환경 생성
python3 -m venv venv
source venv/bin/activate

# 3. 의존성 설치
pip install -r requirements.txt [cite: 1]
```

## 4\. How to Use

### Step 1: Prepare Dataset

[cite\_start]데이터셋을 `ImageFolder` 구조에 맞게 `dataset/` 폴더 내에 배치합니다[cite: 176, 177].

```
dataset/
├── train/
│   ├── water-deer/
│   │   ├── img1.jpg
│   │   └── ...
│   └── roe-deer/
│       ├── img1001.jpg
│       └── ...
└── valid/
├── water-deer/
└── roe-deer/
```

### Step 2: Train Models (Ablation Study)

[cite\_start]`train.py`는 `--contrastive`와 `--oversampling` 플래그를 조합하여 논문의 Ablation Study [cite: 149]를 재현할 수 있습니다.

[cite\_start]학습된 모델과 로그는 `saved_runs/<실험타입>/<비율>/run_X/` 경로에 자동으로 저장됩니다19.

**A. Baseline (기본 CE Loss)**

```bash
python3 train.py \
--train_data_root ./dataset/train \
--classes water-deer roe-deer \
--ratio 1:0.01 \
--save_root ./saved_runs \
--num_epochs 100
```

**B. [cite\_start]Contrastive Loss Only** [cite: 169]

```bash
python3 train.py \
--train_data_root ./dataset/train \
--classes water-deer roe-deer \
--ratio 1:0.01 \
--save_root ./saved_runs \
--num_epochs 100 \
--contrastive
```

**C. [cite\_start]Oversampling Only** [cite: 170]

```bash
python3 train.py \
--train_data_root ./dataset/train \
--classes water-deer roe-deer \
--ratio 1:0.01 \
--save_root ./saved_runs \
--num_epochs 100 \
--oversampling
```

**D. [cite\_start]Proposed (Contrastive + Oversampling)** [cite: 171, 172]

```bash
python3 train.py \
--train_data_root ./dataset/train \
--classes water-deer roe-deer \
--ratio 1:0.01 \
--save_root ./saved_runs \
--num_epochs 100 \
--contrastive \
--oversampling
```

### Step 3: Evaluate Best Models

[cite\_start]`validation.py` 스크립트는 각 실험 타입(`baseline`, `contrastive` 등)의 모든 `run_X` 폴더를 순회하며, `train.log` 파일의 'Train Acc'가 가장 높은 **best model**을 자동으로 찾아내어 평가를 수행합니다[cite: 173, 178].

결과는 터미널에 출력되고, `saved_runs/<실험타입>/result_log.txt` 파일에 요약 저장됩니다.

```bash
# 예시: 'baseline' 실험 전체 평가
python3 validation.py \
--model_root ./saved_runs/baseline \
--test_data_root ./dataset/valid \
--classes water-deer roe-deer
```

## 5\. Results

본 코드를 통해 [그림 3] 및 [표 1]의 실험 결과를 재현할 수 있습니다. [cite\_start]제안하는 방법(Oversample + Contrastive Loss)은 데이터 불균형 비율이 심화되어도(e.g., 0.005) 다른 방법들보다 월등히 높은 정확도를 안정적으로 유지합니다[cite: 129, 147, 153].

> [cite\_start]그림 3: 데이터 불균형 비율에 따른 클래스별 정확도 [cite: 138]

> [cite\_start]표 1: 불균형 데이터 조건에서 방법 별 클래스 정확도 (Ablation Study) [cite: 139, 149]

## 6\. Citation

본 연구가 유용하셨다면, 아래와 같이 인용을 부탁드립니다.

```
이서희, 박가람, 손규원, 도희찬, 양희성, 박혜영. (2025). "Feature Similarity Contrastive Learning 기반 불균형 Fine-Grained 야생동물 이미지 분류 방법". 2025 한국컴퓨터종합학술대회 (KCC 2025).
```

(README.md 여기까지)

-----

### 3\. 업로드 단계 요약

1.  **새 저장소 생성:** 깃헙에서 `KCC-2025`라는 이름으로 Public 저장소를 만듭니다.
2.  **파일 준비:**
* `train.py`, `validation.py`, `test.py`, `utils.py`, `requirements.txt` 파일을 한 폴더에 모읍니다.
* 논문 PDF 파일명을 `KCC_2025_Paper.pdf`로 변경하여 같은 폴더에 둡니다.
* 위에서 제안한 `.gitignore` 내용을 `.gitignore` 파일로 만들어 추가합니다.
* 위에서 제안한 `README.md` 내용을 `README.md` 파일로 만들어 추가합니다.
3.  **데이터 폴더 생성:** `dataset/train`과 `dataset/valid` 폴더를 만듭니다. (내부에 `.gitkeep`이라는 빈 파일을 하나씩 넣어두면 빈 폴더도 깃헙에 업로드할 수 있습니다.)
4.  **업로드:** 이 파일들을 `KCC-2025` 저장소에 업로드합니다.
* `git init`
* `git add .`
* `git commit -m "Initial commit for KCC 2025 paper"`
* `git branch -M main`
* `git remote add origin <저장소_URL.git>`
* `git push -u origin main`

이렇게 하시면 논문 아이디어, 코드, 사용법, 결과가 모두 포함된 아주 훌륭한 연구 저장소가 완성됩니다\!$$
