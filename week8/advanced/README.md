# 📌 추천 질문 생성 LLM 서비스 – 경량화 실험 보고서

## 🧠 프로젝트 개요

사용자의 문맥(Context)에 따라 적절한 **추천 질문**을 생성하는 LLM 서비스를 구축합니다.  
예: RAG 기반 QA 시스템에서 문서를 보고 "이 내용과 관련된 질문"을 자동으로 제안하는 기능.


코드 : 
평가 : after_outputs.json 정성평가   

---

## 🗂️ 데이터셋

- **KorQuAD 1.0** 사용
- `paragraph`와 `question`을 추출하여 `{context: paragraph, question: question}` 형태로 구성
- 초기에는 **140개 샘플**로 실험했으나, 학습 시간이 짧아 **1000개 샘플로 데이터 확장** 가능

---

## ⚙️ 실험 과정 및 경량화 전략

### 1. ❌ 초기 시도: Full Fine-Tuning on Gemma

- **모델**: `gemma-3b`, `gemma-1b`
- **결과**: 보유 GPU 메모리 한계로 **OOM(Out-Of-Memory)** 발생 → 학습 자체 불가

### 2. ✅ LoRA 기반 경량화 시도 (Gemma-1b)

- **모델**: `gemma-1b` + LoRA
- **결과**: 학습 가능해졌으나, 출력 품질이 낮고 질문 생성 성능이 매우 부족

#### ❌ 출력 예시 (문장이 비정상적이거나 무의미함)

```json
{
  "input": "《라라랜드》는 2016년 공개된 미국의 뮤지컬 영화로, 데이미언 셔젤이 감독과 각본을 맡았다...",
  "expected_output": "라라랜드의 각본을 맡은 사람은?",
  "model_output": "라"
}
```

```json
{
  "input": "영생교는 ‘살아 영생’을 주장하는 종교로...",
  "expected_output": "영생교는 타 종교와 다른게 어떤 교리를 표방하는가?",
  "model_output": "영생교는 다음의 교리를 표방하고 있다?\n###\n###\n...(반복)..."
}
```

### 3. ✅ 모델 교체 및 경량화 유지 (Exaone 2.4B + LoRA)

- **모델**: `brainai/exaone-2.4b` + LoRA
- **결과**:
  - **학습 가능** (LoRA 덕분에 2.4B 모델도 처리 가능)
  - **한국어 출력 품질 우수** → 질문 생성 적합

#### ✅ 개선된 출력 예시

```json
{
  "input": "《라라랜드》는 2016년 공개된 미국의 뮤지컬 영화로, 데이미언 셔젤이 감독과 각본을 맡았다...",
  "expected_output": "라라랜드의 각본을 맡은 사람은?",
  "model_output": "라라랜드의 감독은 누구야?"
}
```

```json
{
  "input": "홋카이도의 구리야마 정의 요청으로 개인 야구장을 건립하고, 문화 대상을 수상하였다...",
  "expected_output": "구리야마가 잔디 학회에서 주어지는 문화 대상의 수상자로 선정된 때는?",
  "model_output": "홋카이도에 본거지를 옮긴 야구팀은?"
}
```

---

## 🧪 경량화 성과 요약

| 항목              | Full FT (Gemma 1b) | LoRA (Gemma 1b) | LoRA (Exaone 2.4b) |
|-------------------|--------------------|------------------|---------------------|
| 학습 가능 여부    | ❌ OOM             | ✅ 가능         | ✅ 가능             |
| 한국어 성능       | -                  | 낮음             | 높음                |
| 모델 파라미터 크기| 중간 (1b)          | 중간 (1b)        | 큼 (2.4b)           |
| 메모리 효율       | 낮음               | 높음             | 높음                |
| 데이터 확장 여유  | 없음               | 가능 (1000개)    | 가능 (1000개)       |

---

## ✅ 결론

- **LoRA 적용**을 통해 동일한 GPU 환경에서도 **학습 가능성 확보**
- 작은 모델은 학습은 가능했으나 **출력 품질 미흡**
- **더 큰 모델(Exaone 2.4B)**로 전환하면서도 LoRA 덕분에 메모리 문제 없이 학습 가능
- 그 결과, **질문 생성 품질 향상 + 자원 효율성 확보 + 데이터 확장**이라는 실질적인 이점이 있었다.
