## [8주차] 기본과제 - LoRA rank에 따른 학습 성능 비교해보기

* 학습 코드 :    
* wandb link : https://wandb.ai/cyooon-kim-personal/lora_rank_experiment/workspace?nw=nwusercyooonkim    

---

### loss
<img src="https://github.com/user-attachments/assets/14acec7a-6380-4f9f-8b46-f4bd7756ed4a" width="400"/>

rank에 따라 큰 차이는 없었다.

### 학습 속도   
![image](https://github.com/user-attachments/assets/06848826-3f0f-4754-b211-479f995937a0)

학습속도는 lora_rank에 따라 큰 차이는 없었다.   

### GPU 성능   
![image](https://github.com/user-attachments/assets/51b0a407-5896-42b2-a936-614958f40fd7)

* GPU memory allocated (VRAM 점유율)   
LoRA 파라미터에 따른 VRAM 점유율 증가를 확인할 수 있다.

---

LoRA rank에 따른 학습 성능과 속도에서는 큰 차이가 나타나지 않았다.   
다만, GPU 메모리 사용량은 rank가 높아질수록 점진적으로 증가하는 경향을 보였다.   
따라서 VRAM 여유에 따라 적절한 rank를 선택하는 것이 효율적이다.   
