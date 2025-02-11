# 🔊 오디오 언어모델의 경량 모델링 레서피 탐구

> Audio adapter의 결합 및 사전학습을 통해, 언어모델은 음성/음악/환경음 등의 소리를 이해하고 다양한 downstream task를 수행할 수 있게 되었습니다. VRAM의 크기가 작은 전형적인 디바이스 환경에서는 오디오 언어모델에 대한 경량 모델링이 필수적입니다.
Audio understanding benchmarks에 대한 baseline 모델의 정확도를 유지하면서도, 더 작고 빠른 모델을 만드는 레서피를 디자인 해봅시다.
- 주최 : NOTA, boostcamp AI Tech 7th
- 기간 : 2025.01.10 ~ 2025.02.10

<br>

# 1. Members 👨🏻‍💻👩🏻‍💻

|                         곽기훈                         |                            김민지                            |                         김현기                         |                         이해강                         |                          장희진                          |                        홍유향                        |
|:------------------------------------------------------:|:------------------------------------------------------------:|:------------------------------------------------------:|:--------------------------------------------------------:|:----------------------------------------------------:|:----------------------------------------------------:|
|  <img src="https://github.com/kkh090.png" width="250"> |   <img src="https://github.com/qzzloz.png" width="250">   |  <img src="https://github.com/hyeonrl98.png" width="250">   |  <img src="https://github.com/lazely.png" width="250">   | <img src="https://github.com/heeejini.png" width="250">| <img src="https://github.com/hyanghyanging.png" width="250"> |
| [kkh090](https://github.com/kkh090) | [qzzloz](https://github.com/qzzloz) | [hyeonrl98](https://github.com/hyeonrl98) | [lazely](https://github.com/lazely) | [heeejini](https://github.com/heeejini) | [hyanghyanging](https://github.com/hyanghyanging) |

<br>


#  2. Project Structure 🗂️
```plaintext
📦 level4-cv-finalproject-hackathon-cv-18-lv3/
 ┣ 📂 configs
 ┃ ┣ 📜 train_stage1.yaml
 ┃ ┣ 📜 train_stage2.yaml
 ┃ ┣ 📜 salmonn_eval_config.yaml
 ┃ ┗ 📜 … 
 ┣ 📂 data
 ┣ 📂 demo
 ┣ 📂 eda
 ┣ 📂 wandb
 ┣ 📂 LLMPruner
 ┣ 📂 models
 ┃ ┣ 📂 beats
 ┃ ┃ ┗ 📜 … 
 ┃ ┣ 📜 salmonn.py
 ┃ ┗ 📜 utils.py
 ┣ 📂 scripts
 ┃ ┣ 📜 stage1_run.sh
 ┃ ┣ 📜 stage2_run.sh
 ┃ ┣ 📜 run_eval_efficiency.sh
 ┃ ┗ 📜 … 
 ┣ 📂 utils
 ┃ ┣ 📜 utils.py
 ┃ ┣ 📜 dist_utils.py
 ┃ ┣ 📜 salmonn_utils.py
 ┃ ┣ 📜 logger.py
 ┃ ┣ 📜 metrics.py
 ┃ ┗ 📜 runner.py
 ┣ 📜 config.py
 ┣ 📜 dataset.py
 ┣ 📜 optims.py
 ┣ 📜 train.py
 ┣ 📜 eval.py
 ┣ 📜 evaluate_salmonn.py
 ┣ 📜 evaluate_efficiency_salmonn.py
 ┗ 📜 … 
```
<br>
<br>



# 3. Requirements 💻
1. Conda 가상환경 생성
```
conda create -n <가상환경명> python=3.9.17
conda activate <가상환경명>
```
2. requirements 설치
```
bash scripts/requirements.sh
```
3. 사전 학습 모델 체크포인트 다운로드
- llama-1b https://huggingface.co/meta-llama/Llama-3.2-1B
- whisper-medium https://huggingface.co/openai/whisper-medium


<br>

# 4. Run 🏃🏻
## Train
```
bash scripts/stage1_run.sh
bash scripts/stage2_run.sh
```

## Inference
```
bash scripts/run_submission_asr.sh
```
## Latency 
```
bash scripts/run_eval_efficiency.sh
```
<br>

# 5. Reference 🔗

