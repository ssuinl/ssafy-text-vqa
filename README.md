# 텍스트 이미지 VQA

SSAFY 16기 AI 챌린지 | 4인 팀 (팀장) | **공식 최종 리더보드 전체 8위** (참가자 약 1,000명)

Public LB 0.95204 → 0.97438 · 검증 654/672 (97.32%)

담당: 모델 학습, 앙상블 설계 및 성능 검증을 주도했습니다.

간판·메뉴판·포스터처럼 **글자가 들어 있는 이미지**와 한국어 질문을 보고, 4개 선택지(a~d) 중 정답을 고르는 Visual Question Answering 과제입니다. 서로 다른 계열의 비전-언어 모델(Qwen2.5-VL, Qwen3-VL, Gemma 4, Qwen3.5)을 파인튜닝하고, 각 모델이 낸 선택지별 확률을 계열 단위로 평균해 최종 답을 정했습니다.

## 목적

- 이미지 속 텍스트를 읽고(OCR 성격) 질문의 의도까지 이해해야 하는 한국어 멀티모달 문제에서 정확도 개선
- 리더보드에 과적합하지 않도록 **검증 기준을 먼저 정하고**, 그 기준을 통과한 변경만 채택하는 실험 진행
- Colab(A100 40GB) 한 장의 제약 안에서 27~31B급 대형 모델까지 활용

## 데이터

| 구분 | 문항 수 | 사용 방법 |
|---|---|---|
| train | 6,714 | 라벨 정제로 15문항 제외 → seed 43으로 **valid 672문항** 분리, 나머지 6,027문항으로 학습 |
| dev | 2,026 | 교육생 응답 데이터 → 8B 모델 학습에 pseudo-label로만 사용 |
| test | 6,714 | 제출용 |

- 각 문항: 이미지 1장 + 한국어 질문 + 선택지 4개(a~d), 정답 1개
- 대회 데이터는 재배포할 수 없어 이 레포에는 포함하지 않았습니다. 노트북은 Google Drive에 있는 대회 zip을 Colab 로컬로 복사해 푸는 구조입니다.
- **누수 발견**: 라벨 정제 중 새로 나눈 분할에서 valid 672문항 중 422문항이 학습용에도 섞여 있던 것을 찾아, 2단계부터 valid를 학습에서 완전히 제외하고 코드에서 겹침 0을 assert로 확인했습니다.

## 방법

### 핵심 설계

1. **답을 생성하지 않고 확률을 읽기** — 마지막 위치의 로짓에서 a/b/c/d 토큰(대소문자·공백 변형 포함) 확률만 모아 정규화합니다. 출력 형식 오류가 없고, 모델끼리 확률을 바로 평균할 수 있습니다.
2. **정답 글자 토큰에만 loss** — 질문·이미지 토큰은 `-100`으로 가리고 정답 기호 한 글자만 학습합니다. 학습 때 **선택지 순서를 무작위로 섞어** 위치 편향을 줄였습니다.
3. **계열별 그룹 앙상블** — 같은 계열의 변형은 먼저 평균해 한 그룹으로 만들고, 그룹끼리는 동일 비율로 평균합니다. 최종 네 계열의 가중치는 동일하게 고정했습니다. 단, Qwen2.5-VL 계열 내부에는 0.52/0.48 가중치를 사용했습니다.
4. **채택 기준** — 검증 672문항에서 4문항 이상 개선되고 부트스트랩 1,000회 중 80% 이상 우세할 때 채택하는 기준을 사용했습니다. 이는 당시 실험의 채택 기준이며 통계적 유의성 검정 결과는 아닙니다. 부트스트랩 판정 코드는 현재 저장소에 포함되어 있지 않습니다.

### 모델 교체 3단계

| 단계 | 모델 | 내용 |
|---|---|---|
| ① | Qwen2.5-VL 3B · 7B | 4bit LoRA로 valid 96.1%까지 도달했으나 이후 60여 개 실험이 ±2문항에서 정체. 오답 분석 결과 오답의 73%가 확신도 하위 12% 문항에 몰려 있어 모델 자체를 바꾸기로 결정 |
| ② | Qwen3-VL-8B | bf16 LoRA, 선택지 섞기 증강 + dev pseudo-label → LB 0.96604 |
| ③ | Gemma 4 31B · Qwen3.5-27B | A100 40GB에서 4bit 제로샷·QLoRA, 그룹 앙상블에 추가 → LB 0.97438 |

### 최종 조합

```
그룹 1  Qwen2.5-VL   = 0.52 × 3B(01) + 0.48 × 7B(02)              valid 646
그룹 2  Qwen3-VL-8B LoRA (03)                                      valid 650
그룹 3  Gemma 4 31B  = (QLoRA(07) + 제로샷 1120토큰(05)) / 2       valid 647 · 648
그룹 4  Qwen3.5-27B  = (QLoRA(06) + 고해상도 추론(06b)) / 2        valid 649 · 649
최종    = 네 그룹 동일 평균 → argmax                               valid 654/672
```

### 핵심 코드

```python
# ① 정답 글자 토큰에만 loss + 선택지 순서 섞기
order = random.sample(range(4), 4)
opts  = [opts[i] for i in order]
gold  = "abcd"[order.index(gold_idx)]
enc   = processor(text=[chat(img, prompt(q, opts)) + gold], images=[img], return_tensors="pt")
labels = torch.full_like(enc["input_ids"], -100)
labels[0, -1] = enc["input_ids"][0, -1]

# ② 생성 없이 a~d 확률 읽기
lg = model(**inp, logits_to_keep=1).logits[0, -1]
p  = softmax(stack([logsumexp(lg[ids], 0) for ids in CHOICE_IDS]), 0)

# ③ 그룹별 평균 → 동일 비율 앙상블
groups = [0.52*P["3b"] + 0.48*P["7b"], P["8b"],
          (P["gemma_qlora"] + P["gemma1120"]) / 2,
          (P["qwen35"] + P["qwen35_px1536"]) / 2]
answer = ["abcd"[i] for i in (sum(groups) / len(groups)).argmax(1)]
```

## 결과

### 리더보드 변화

| 조합 | Public LB |
|---|---|
| Qwen2.5-VL 3B + 7B | 0.95204 |
| + Qwen3-VL-8B | 0.96604 |
| + Gemma 4 제로샷 (560+280 평균) | 0.96782 |
| Gemma 그룹을 QLoRA로 교체 | 0.96872 |
| + Qwen3.5-27B QLoRA | 0.97289 |
| Gemma 그룹에 1120토큰 제로샷 평균 | 0.97408 |
| **Qwen3.5 그룹에 고해상도 추론 평균 (최종)** | **0.97438** |

### 효과가 없었던 시도

| 시도 | valid | Public LB | 해석 |
|---|---|---|---|
| 저신뢰 문항 멀티크롭 재추론 | +2 | −0.21%p | 채택 기준 미충족 |
| 순위 투표(Borda) | +2 | −0.24%p | 확신의 크기 정보를 버림 |
| 최종 조합에 확신도 보정 | +1 | −0.18%p | valid에 맞춘 온도가 test에선 어긋남 |
| 8B seed 변경 추가 | ±0 | −0.12%p | 같은 모델 변형은 오답이 겹침 |

검증에서 2문항 개선된 시도도 Public LB에서는 점수가 하락했습니다.

### 실험에서 확인한 점

- **해상도가 가장 직접적인 변수**: Gemma 이미지 토큰 280 → 560 → 1120에서 valid 635 → 645 → 648
- **다른 계열을 섞을수록 효과가 큼**: 오답 겹침(Jaccard) Qwen2.5 3B↔7B 0.44, Qwen2.5 7B↔Gemma 0.31
- **추론 속도**: T4에서 4bit 양자화 추론이 문항당 30초 이상 걸리던 문제를 bf16 전환으로 0.3초로 단축
- 최종 valid 오답 18개 중 8개는 모든 모델이 틀린 문항 → 앙상블로 넘기 어려운 한계

## 저장소 구성

```
notebooks/
  01_qwen25vl3b.ipynb            Qwen2.5-VL-3B 4bit LoRA, 3단계 학습 + 1280px 추론
  02_qwen25vl7b.ipynb            Qwen2.5-VL-7B 4bit LoRA, 2단계 학습
  03_qwen3vl8b.ipynb             Qwen3-VL-8B bf16 LoRA (선택지 섞기 + dev pseudo-label)
  05_gemma4_zeroshot.ipynb       Gemma 4 31B 4bit 제로샷 (이미지 토큰 280/560/1120)
  06_qwen35_27b_qlora.ipynb      Qwen3.5-27B 4bit QLoRA
  06b_qwen35_px1536_infer.ipynb  06의 LoRA로 고해상도(약 1,536토큰) 추론
  07_gemma4_31b_qlora.ipynb      Gemma 4 31B 4bit QLoRA
  04_final_ensemble.ipynb        확률 CSV → 그룹 평균 → 제출 파일 (CPU)
experiments/                     최종 파이프라인 밖에서 시도한 실험 (출력 제거)
  00_data_analysis.ipynb                     데이터 탐색 (크기·컬럼·정답 분포)
  EXP-101_selective_reinfer_perm_crop.ipynb  저신뢰 78문항만 선지 순열 × 멀티크롭으로 재추론
  EXP-105_internvl3_8b_lora.ipynb            다른 계열(InternVL3-8B) LoRA로 다양성 확보 시도
  EXP-112_submission_candidates.ipynb        미리 정한 후보(단독·다수결·동일 평균)만 만들어 제출 비교
  EXP-126_temperature_calibration.ipynb      모델별 온도 보정 후 로그 평균
  EXP-128_gemma4_shift_tta.ipynb             Gemma QLoRA 선지 순서 이동 TTA
docs/
  solution_description.pdf       솔루션 설명서 (5쪽)
  presentation.pdf               발표 자료 (15쪽)
```

`experiments/`의 노트북은 대부분 **채택되지 않은 시도**입니다. 결과는 위 「효과가 없었던 시도」 표에 정리했고, 원본 실험 경로(`experiment_data/`)의 확률 파일을 읽기 때문에 그대로 다시 실행되지는 않습니다. 어떤 가설로 무엇을 시도했는지 보여 주려고 함께 올렸습니다.

## 실행 방법

- 환경: Google Colab Pro, A100 40GB · Python 3.13, PyTorch 2.11, transformers 5.x, peft 0.21, bitsandbytes 0.50 (06은 flash-linear-attention 추가)
- 각 노트북 첫 셀에서 패키지를 설치한 뒤 **런타임 → 세션 다시 시작**, 이후 위에서부터 실행
- 노트북의 기본 경로는 `/content/drive/MyDrive/ssafy/ai-challenge`입니다. 대회 데이터 zip과 `split/valid_split_seed43.csv`를 이 위치에 두거나 경로를 바꿔 주세요.
- 01~03, 05~07은 서로 독립(06b만 06 다음) → 마지막에 04 실행
- 06·07의 학습량은 속도 측정 셀이 시간 예산에 맞춰 정합니다. 원본과 같게 하려면 `TRAIN_N = 3784`(06), `TRAIN_N = 2560`(07)으로 고정하세요.
- GPU 연산의 비결정성 때문에 재실행 시 valid가 1~3문항 달라질 수 있습니다.

## 검증의 한계

- 동일한 valid 672문항을 모델 및 앙상블 선택에 반복 사용했으므로 97.32%는 독립적인 최종 평가 성능이 아닙니다.
- 위 Public LB 점수는 당시 제출 기록입니다. 최종 순위(전체 8위)는 대회 공식 최종 리더보드 기준입니다.
- 원본 확률 CSV와 대회 데이터가 없어 현재 저장소만으로 기록된 점수를 독립적으로 재현할 수 없습니다.
