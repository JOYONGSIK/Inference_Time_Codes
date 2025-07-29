## Inference 환경 및 측정 방법

### 환경 정보
- UNIST 서버에서 진행 (GPU: RTX 3090)
- Python 라이브러리 버전:
  - `transformers==4.46.3`
  - `torch==2.0.1+cu117`
- CUDA 버전: 12.3 (`.bashrc`에서 확인했음.)

### LLaMA 8B 모델 사용 시 주의 사항
- Hugging Face Token이 필요함
- 토큰은 llama 8b code,  7번째 줄에 입력

### 리소스 모니터링 방법
- watch -n 0.5을 사용해서 확인했습니다.
- **GPU 메모리 사용량 확인할 때**
    ```bash
    nvidia-smi
    watch -n 0.5 nvidia-smi
    ```
- **Power Consumption 체크하는 법**
    ```bash 
    nvidia-smi -q -d POWER
    watch -n 0.5 nvidia-smi -q -d POWER
    ```

### 진행한 실험 프로세스
1.	for문을 100번 반복했을 때 전체 소요 시간 측정
2.	Inference 과정에서 GPU 메모리 확인
3.	Inference 과정 중 Power Consumption(Power Draw) 확인
4.	CLIP+OPT / CLIP+Llama 코드 실행 후 메모리, Power Consumption(Power Draw) 확인 
5.	CLIP, Llama, OPT에서 time.sleep(1) 넣고 Power Consumption이 어떻게 변경되는지 확인

- CLIP+OPT / CLIP+Llama 한번에 돌릴 때, 터미널 2개 켜놓고, CUDA_VISIBLE_DEVICES=0 으로 GPU 하나에 적용시켜서 돌렸었음.
- merge_clip_llama.py, merge_clip_opt.py 파일 작성 후 코드 돌아가는 것 까지 확인 완료. 
