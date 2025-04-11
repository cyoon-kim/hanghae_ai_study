# MNLI Fine-Tuning Report

---

## 1. 선택한 Task

- **Task:** Multi-genre Natural Language Inference (MNLI)  
- **문제 설명:**  
  두 개의 문장(전제(premise)와 가설(hypothesis))이 주어졌을 때, 두 문장이 서로 논리적으로 연결되어 있는지, 모순되는지, 혹은 관련이 없는지를 판별하는 문제입니다.  
- **데이터셋:**  
  - 학습용: train.csv (392,702행)  
  - 검증용: validation_matched.csv (9,815행)  
- **데이터셋 칼럼:**
  - 아래 3가지 칼럼만 사용합니다.   
  - `premise`: 전제 문장  
  - `hypothesis`: 가설 문장  
  - `label`: 레이블 (0: Entailment, 1: Neutral, 2: Contradiction)  

---

## 2. 모델 설계 및 입력/출력 형태

### 모델 구조

- **Pre-trained Model:**  
  - [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) 모델을 사용하여 사전 학습된 언어 표현을 활용함.  
  - 모델의 파라미터는 학습 시 동결(freeze)하였습니다.

- **MNLIClassifier:**  
  - 사전 학습된 모델의 출력(hidden state)의 첫 번째 토큰([CLS] 위치)를 추출하여 분류에 활용.  
  - 추출된 hidden state의 차원(`hidden_size`)을 기반으로, 3개의 클래스로 매핑하는 선형 계층(`nn.Linear`)을 추가함.  
  - 최종 출력은 3개의 로짓(logits)으로, 각각 Entailment, Neutral, Contradiction에 해당함.

### 입력 및 전처리

- **입력 데이터:**  
  - 전제(premise)와 가설(hypothesis) 문장 쌍 (pair)  
  - 토크나이저를 통해 토큰화되며, padding, truncation 처리가 이루어짐  
  - DataLoader의 `collate_fn`에서 두 문장을 하나의 문장 쌍으로 결합하여 모델 입력 형식에 맞추어 변환함.  
  - 최종 입력은 `input_ids`와 `attention_mask` (추가로 `labels`가 함께 반환됨).

### 출력

- **출력:**  
  - 모델의 출력은 3차원의 로짓 텐서로, 각 차원은 각각의 레이블(Entailment, Neutral, Contradiction)에 대한 예측 점수를 나타냄.  
  - 손실 함수는 CrossEntropyLoss를 사용하여 각 배치별 예측과 실제 레이블 간의 차이를 계산함.

---

## 3. Fine-Tuning 및 성능 점검

### Fine-Tuning 손실 곡선(Fine-tuning Loss Curve)

![image](https://github.com/user-attachments/assets/e5b7c144-517c-428e-acf5-0b614bcfe69f)

- **분석**
  loss는 줄어드는 모습을 보이나, acc는 나아지지 않는 모습을 보입니다.   
  왔다갔다 하는 모습도 보였습니다.
  
### learning rate 변경

- **비교 내용:**
  - 성능 개선을 위해 하이퍼파라미터를 값을 조정하던 중 learning rate가 큰 영향을 미치는 것을 확인할 수 있었습니다.

![image](https://github.com/user-attachments/assets/5725857c-4d68-447b-bc9c-7e75c7bd995a)

  - **learning rate를 낮춰야 성능이 잘 나오는 이유**
    - 사전 학습된 지식 보전을 위해
    - 파라미터가 이미 좋은 위치에 있기 때문에
    - lr이 크면 loss가 널뛰기하거나 발산할 위험이 크기 때문에
  
---
