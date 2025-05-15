## 🧪 [7주차] 심화과제: 자신만의 text 파일로 LLM Instruction-tuning 해보기

### 📌 실험 목적   
- `google/gemma-3-1b-it` 모델을 대상으로 **한국어 질문 생성 task**에 대해 **LoRA 기반 미세조정(Fine-tuning)** 가능성 검토   
- 소량의 커스텀 instruction 데이터셋(`corpus.json`, 140개 샘플)을 활용해 성능 확인   

---

### ⚙️ 환경 및 설정

- **모델**: `google/gemma-3-1b-it`
- **튜닝 방식**: [PEFT](https://github.com/huggingface/peft) + LoRA (GPU에서 full finetune 시도시 OOM 발생생)
- **GPU**: 14GB VRAM
- **기술 스택**:
  - `transformers`, `datasets`, `peft`, `bitsandbytes`, `wandb`
- **LoRA 설정**:
  - `target_modules`: `["q_proj", "k_proj", "v_proj", "o_proj"]`
  - `r=8`, `alpha=16`, `dropout=0.05`
  - `fp16`, `gradient_checkpointing` 사용

---

### 🧩 데이터셋 요약

- 자체 구축한 140개 instruction 샘플
- 각 샘플은 다음 형태:
```json
{
  "instruction": "다음 문단을 읽고 질문을 생성하세요.",
  "input": "<문단 내용>",
  "output": "<질문>"
}
```
KorQuAD 2.0 : https://github.com/korquad/korquad.github.io/blob/master/dataset/KorQuAD_2.0/dev/KorQuAD_2.0_dev_01.zip

---

### 📈 결과 요약

- **LoRA 적용**하여 OOM 발생 없이 학습은 성공하였지만 모델이 제대로 학습하지는 못함
- 데이터 셋이 너무 적기 때문으로 보임. 관련 데이터는 많기 때문에 숫자를 쉽게 늘릴 수 있고 해당 변경점을 적용해야 함

---

### 관련 링크
* 데이터 생성 :
* 모델 학습 : 
* 학습 전/후 데이터   
- 학습 전 :    
- 학습 후 :    

---

### wandb

