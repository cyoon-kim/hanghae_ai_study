# 보고서: Named Entity Recognition (NER) 실습 보고서 📚

실습 파일 : https://github.com/cyoon-kim/hanghae_ai_study/blob/main/week3/advanced/NER/NER.ipynb   

---

## 1. 선택한 Task 📝

- **Task:** Named Entity Recognition (NER)
- **설명:**  
  텍스트 내에서 의미 있는 개체(예: 인물, 장소, 기관 등)를 인식하는 NER 스크를 선택하였습니다.
- **데이터셋**  
  https://www.kaggle.com/datasets/debasisdotcom/name-entity-recognition-ner-dataset   
  위 데이터 셋을 train 0.8 test 0.2 로 나누어 사용하였습니다.

---

## 2. 학습 데이터 전처리 📂

- **데이터 전처리**
  sentence 별로 Word와 Tag를 그룹화 합니다.   
  Tag의 값을 label화 하여 반영합니다.   
  label 값을 subword token을 반영한 labels 로 변경이 필요합니다.

---

## 3. 모델 설계 및 입출력 형태 🧑‍💻

- **base model:**
  모델별로 비교하고 싶어 3가지 모델을 사용하였습니다.    
  - **distilBERT**
    BERT 기반의 모델, knowledge distillation 기법을 통해 경량화   
    https://huggingface.co/distilbert/distilbert-base-cased
  - **RoBERTa**
    BERT 모델 기반으로 더 많은 데이터와 더 많은 학습 시간으로 pretrained 된 모델
    https://huggingface.co/FacebookAI/roberta-base
  - **BERT**
    범용적으로 많이 쓰이는 모델 (기준으로 삼기 좋을 것 같아 선택)
    https://huggingface.co/google-bert/bert-base-cased

- **NERClassification**
  - encoder
    pretrained 모델을 그대로 사용합니다.   
    이 parameter는 freeze 합니다.
  - classifier
    tag의 값이 unique한게 17개이기 때문에 (encoder.hidden_size, 17) 로 구성됩니다.   

- **입력 형태:**  
  - 모델의 입력은 텍스트의 토큰화된 형태입니다.  

- **출력 형태:**  
  - 모델의 출력은 각 토큰별로 NER 라벨(개체명 태그)에 대한 예측 결과입니다. (17개)  
  - label index를 나타내는 tensor로 return됩니다.

---

## 4. Fine-tuning 결과 및 Loss Curve 비교 📊

- **Pre-trained 모델 Fine-tuning 시 Loss Curve:**  
  - **결과:**  
    사전 학습된 모델을 fine-tuning할 경우 초기 학습 단계에서 loss 값이 급격히 감소하는 경향을 보였으며, 이후 안정적으로 수렴하는 모습을 확인할 수 있었습니다.
    
<p align="center">
  <img src="https://github.com/user-attachments/assets/22e7d211-0d03-46fd-9876-20caae30ef54" width="30%" />
  <img src="https://github.com/user-attachments/assets/c1490656-f7a2-4607-bbb7-230b89fb8b01" width="30%" />
  <img src="https://github.com/user-attachments/assets/716cc1b7-2a94-4f6f-ad3a-9bead12e8d08" width="30%" />
</p>

+) 코드 이슈로 name이 잘못 들어갔지만 distilBERT, roBERT, BERT 순서의 loss graph입니다.

- **성능 비교 지표**
  - Accuracy
    전체 토큰 중 맞춘 비율
  - Macro avg
    클래스별 F1의 단순 평균 F1_score.mean()
  - **weighted avg**
    클래스별 F1의 가중 평균 ((F1_score * weight).mean())   
    label "O" f1-score : 0.988 / label "I-art" f1-score : 0.101
    불균형 데이터일땐 weighted avg를 사용하므로 이 지표를 가지고 성능 비교를 하기로 결정했습니다.

- **Pre-trained 모델과 finetuning 모델의 NER 성능 비교:**

<p align="center">
  <img src="https://github.com/user-attachments/assets/2cd6d180-de8e-4c6e-afc8-948663d0655f" width="30%" />
  <img src="https://github.com/user-attachments/assets/227c79c3-f6ca-42f3-a3c4-21316febc781" width="30%" />
  <img src="https://github.com/user-attachments/assets/5dda9c76-39ce-423d-96e1-2f99dff94fea" width="30%" />
</p>

  - **결과**
    세 모델 모두 베이스 모델의 성능보다 대폭 성능이 올라간 걸 확인할 수 있었고, 세 모델 모두 비슷한 성능이 나온걸 확인할 수 있었습니다.
---
