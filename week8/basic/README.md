## [8주차] 기본과제 - LoRA rank에 따른 학습 성능 비교해보기

* 학습 코드 :    
* wandb link :    

---

### loss
rank에 따라 큰 차이는 없었다.

### 학습 속도   
학습속도는 lora_rank에 따라 큰 차이는 없었다.   

### 메모리(RAM) 점유율   
VRAM 뿐만 아니라 RAM도 rank에 따라 점유율이 달라진다.

### GPU 성능   

* GPU memory allocated (VRAM 점유율)   
LoRA 파라미터에 따른 VRAM 점유율 증가를 확인할 수 있다.

* GPU Utilization   
모두 GPU를 풀가동하고 있는걸 확인할 수 있다.