> #### 📌 과제 요구 사항
> - [x] AG_News dataset 준비
> 	- Huggingface dataset의 `fancyzhx/ag_news`를 load
> 	- `collate_fn` 함수에 다음 수정사항들을 반영
>    - Truncation과 관련된 부분들을 삭제
> - [x] Classifier output, loss function, accuracy function 변경
> 	- 뉴스 기사 분류 문제는 binary classification이 아닌 일반적인 classification 문제입니다. MNIST 과제에서 했던 것 처럼 `nn.CrossEntropyLoss` 를 추가하고 `TextClassifier`의 출력 차원을 잘 조정하여 task를 풀 수 있도록 수정
> 	- 그리고 정확도를 재는 `accuracy` 함수도 classification에 맞춰 수정
> - [x]  학습 결과 report
>     - DistilBERT 실습과 같이 매 epoch 마다의 train loss를 출력하고 최종 모델의 test accuracy를 report 첨부


# [3주차] 기본과제 - DistilBERT를 활용한 뉴스 기사 분류

본 과제는 pre-trained된 DistilBERT 모델을 사용하여 AG_News 데이터셋의 뉴스 기사를 분류하는 모델을 구현하고 다양한 실험을 진행한 내용을 담고 있습니다.

## 📌 과제 목표
- Pre-trained DistilBERT를 뉴스 기사 분류 문제에 적용
- Tokenizer 로딩 및 데이터 전처리
- 모델 fine-tuning 및 평가


## 🗃️ 데이터셋

### AG_News Dataset
뉴스 기사의 제목과 내용을 기반으로 텍스트 분류를 수행하는 데이터셋입니다.

#### 데이터 구성 (Base는 5%만 사용)
| 구분   | 데이터 개수 |
|--------|------------|
| train  | 6000       |
| test   | 380        |

#### 클래스 구성
| Index | Category  |
|-------|-----------|
| 0     | World     |
| 1     | Sports    |
| 2     | Business  |
| 3     | Sci/Tech  |

데이터 예시:
```json
{
  "text": "Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, Wall Street's dwindling\band of ultra-cynics, are seeing green again.",
  "label": 2
}
```

## 🛠️ Tokenizer 및 데이터 전처리
- DistilBERT tokenizer 사용
- Tokenization 및 padding 작업 수행 (자동 padding 적용)
- 배치 크기(batch size)는 64로 설정

## 🔧 모델 구조
- Pre-trained DistilBERT 모델 사용 (`distilbert-base-uncased`)
- 분류를 위한 추가적인 선형 레이어(nn.Linear)를 추가하여 클래스(4개)를 분류
- DistilBERT의 encoder 파라미터는 동결 (freeze)하여 학습에서 제외함

## 📈 학습 및 평가
Accuracy over epoch
![image](https://github.com/user-attachments/assets/0b64e412-0491-4e04-a80a-735c49d7afda)

Loss over epoch
![image](https://github.com/user-attachments/assets/d75316da-6303-42f8-9ae8-484e1219f869)

- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 64
- Epoch: 5
- 평가 지표: 정확도(accuracy)

## 📊 추가 실험 및 결과 분석
노트북에서 진행된 추가 실험과 결과는 다음과 같습니다:

### 🧪 실험 1: clean text, weight decay 설정
- 소문자 특수문자 여러 공백 등 data 전처리
```python
def clean_text(text):
    text = text.lower()  # 소문자 통일
    text = re.sub(r'\s+', ' ', text)  # 여러 공백 → 하나
    text = re.sub(r'[^]*\]', '', text)  # 괄호 내용 제거
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # URL 제거
    text = re.sub(r'[^a-z0-9.,!?\'\" ]+', '', text)  # 특수문자 제거
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
```
- weight decay 설정

#### 실험 1 결과
![image](https://github.com/user-attachments/assets/e3b938f5-9c9d-4d90-b6cf-d5db3fe815e6)
test data 기준 성능이 더 올라가는 것을 확인할 수 있었습니다.

### 🧪 실험 2: business augmentation
![image](https://github.com/user-attachments/assets/3735118f-a1b8-4009-b7a2-7b67c7f05081)
label 당 acc를 측정하면 business label에 대한 성능이 떨어지는 것을 볼 수 있습니다.

```python
full_ds = load_dataset("fancyzhx/ag_news", split="train")
business_all = full_ds.filter(lambda x:x['label']==2)

n = len(business_all)

start = int(0.95 * n)
extra_business = business_all.select(range(start, n))

print(f"extra business data count is {len(extra_business)}")

augmented_train = concatenate_datasets([all_train_ds, extra_business]).shuffle(seed=42)
```
따라서 business를 label로 가진 30000개의 데이터 중 5%인 1500개를 train_ds에 더해주어 business 를 조금 더 학습하도록 추가했습니다.

#### 실험 2 결과

|Class|	Before|	After|	변화량 (Δ)|
|--|--|--|--|
|Business|	0.7571|	0.8571	|+0.1000 ✅|
|Sci/Tech|	0.8600|	0.7800|	−0.0800 ❌|
|Sports|	0.9381	|0.9381|	+0.0000 🟰|
|World	|0.8247	|0.8351|	+0.0103 ⬆️|

- Business 개선 business 데이터 추가는 성능에 영향을 줌
- Sci/Tech 성능 하락 sci/tech와 business를 헷갈려하는 경향이 보임
- World 약간 상승 Sport 그대로 business 데이터를 추가한건 두 label에는 거의 영향 없음
- 결론적으로 business와 sci/Tech를 모델이 헷갈려 한다는 분석이 나왔습니다.

![image](https://github.com/user-attachments/assets/8c4918ad-e949-4986-927f-ec0977731e01)
- 실제 label값에 따라 어떤 pred가 선택되었는지 찍어봐도 둘을 헷갈려하는 것을 볼 수 있었습니다.
