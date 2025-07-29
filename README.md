## Inference 환경 및 측정 방법

### Python Install 
- 아래 버전은 추천이지, 꼭 따를 필요 없음 
  - ```pip install transformers==4.37.2```
  - ```pip install torch==2.1.2``` 

### 아래 스텝 진행 
1. ```python run_clip_opt.py``` 실행
2. 위 파일이 돌아가는 중에, 아래 GPU 메모리 확인 라인 실행하여 스샷
3. 위 파일이 돌아가는 중에, 아랴 전력 소모(Power Draw) 확인 라인 실행하여 스샷 

### GPU 메모리 확인 방법 
```watch -n 0.5 nvidia-smi``` 

### 전력 소모(Power Draw) 확인
```watch -n 0.5 nvidia-smi -q -d POWER```
위 명령어는 우분투를 위한 것이므로, Jetson은 조금 다를 수 있음
