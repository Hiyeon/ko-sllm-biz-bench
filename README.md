# 🤖 ko-sllm-biz-bench

**Korean business task benchmark for local open-source sLLMs (≤ 9B) via Ollama.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-local-black?logo=ollama)
![License](https://img.shields.io/badge/license-MIT-green)
![Models](https://img.shields.io/badge/models-14-orange)
![Tasks](https://img.shields.io/badge/tasks-4%20Korean%20biz-red)

Ollama로 **9B 이하 경량 sLLM 14개**를 로컬 실행하고,
**한국어 실무 4가지 Task**를 규칙 기반 + Gemini-as-a-Judge 2단계 채점으로 자동 평가합니다.

> 🇰🇷 평가 데이터셋(`eval_cases.json`)은 한국 실무 환경을 직접 반영해 제작된 **한국어 전용** 벤치마크입니다.

---

## 📊 Benchmark Results

> 점수 범위: 0~1 · 형식: mean ± std (20문항 반복 평가) · 환경: Ollama + GGUF, 로컬 실행

| 순위 | 모델 | 파라미터 | Task1<br>JSON 추출 | Task2<br>일정 조율 | Task3<br>이메일 요약 | Task4<br>RAG 환각 | **종합** |
|:---:|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Qwen2.5-7B** | 7B | 0.50±0.21 | **0.48**±0.29 | 0.79±0.20 | 0.94±0.19 | **0.68**±0.22 |
| 🥈 | **Gemma-3-4B** | 4B | 0.51±0.15 | 0.44±0.28 | 0.72±0.20 | **0.97**±0.14 | **0.66**±0.19 |
| 🥉 | **EXAONE-3.5-7.8B** | 7.8B | 0.47±0.20 | 0.42±0.27 | 0.72±0.19 | 0.94±0.19 | **0.64**±0.21 |
| 4 | Llama-3.1-8B | 8B | 0.43±0.20 | 0.33±0.22 | **0.80**±0.21 | **0.97**±0.14 | 0.63±0.19 |
| 5 | Gemma-2-2B | 2B | 0.45±0.20 | 0.30±0.19 | 0.70±0.20 | **0.97**±0.15 | 0.61±0.18 |
| 6 | Qwen2.5-3B | 3B | 0.46±0.21 | 0.35±0.23 | 0.73±0.22 | 0.72±0.25 | 0.56±0.23 |
| 7 | Gemma-3-1B | 1B | 0.42±0.14 | 0.23±0.08 | 0.64±0.24 | 0.91±0.25 | 0.55±0.18 |
| 8 | Gemma-2-9B | 9B | **0.64**±0.41 | 0.00±0.00 | 0.61±0.43 | 0.92±0.27 | 0.54±0.28 |
| 9 | Phi-4-Mini | 3.8B | 0.38±0.12 | 0.30±0.12 | 0.57±0.22 | 0.76±0.31 | 0.50±0.19 |
| 10 | Llama-3.2-3B | 3B | 0.41±0.17 | 0.28±0.15 | 0.62±0.24 | 0.62±0.37 | 0.48±0.23 |
| 11 | Llama-3.2-1B | 1B | 0.33±0.16 | 0.24±0.09 | 0.50±0.20 | 0.69±0.35 | 0.44±0.20 |
| 12 | Qwen2.5-1.5B | 1.5B | 0.43±0.15 | 0.26±0.15 | 0.52±0.16 | 0.55±0.33 | 0.44±0.20 |
| 13 | DeepSeek-R1-7B | 7B | 0.42±0.21 | 0.28±0.20 | 0.44±0.16 | 0.47±0.23 | 0.40±0.20 |
| 14 | Qwen2.5-0.5B | 0.5B | 0.39±0.15 | 0.23±0.08 | 0.51±0.24 | 0.47±0.33 | 0.40±0.20 |

### 💡 Key Findings

- **Gemma-3-4B (4B)** — 파라미터 대비 최고 효율. RAG 환각 통제 0.97로 전체 1위권. 경량 환경 1순위 추천.
- **Qwen2.5-7B (7B)** — 4개 Task 균형 최상. 특히 일정 조율(복합 추론) Task에서 유일하게 0.48 달성.
- **EXAONE-3.5-7.8B** — 한국어 특화 사전학습의 효과. 맥락 이해가 필요한 Task에서 두드러짐.
- **DeepSeek-R1-7B** — 7B임에도 14위. 수학/코딩 추론 특화 모델은 한국어 실무 Task에 최적화되어 있지 않음.
- **Task 2 (일정 조율)** — 전 모델에서 가장 낮은 점수. 복합 제약 논리 처리는 9B 미만 모델의 공통 약점.
- **Gemma-2-9B** — Task2 0.00, std 0.41~0.43. 9B임에도 불안정. Ollama 기본 시스템 프롬프트 처리 방식 이슈 추정.

---

## ✨ Features

- **한국어 실무 특화** — 회의록 처리, 일정 조율, 이메일 요약, 사내 규정 답변 등 국내 업무 맥락 반영
- **직접 제작한 평가 데이터셋** — Task당 20문항(난이도 하/중/상/함정), 총 80문항을 한국어로 직접 작성
- **경량 모델 집중 비교** — 0.5B~9B 구간 14개 모델, 실제 로컬 배포 가능한 크기만 평가
- **Ollama 기반 공정 비교** — 동일 추론 엔진 + GGUF 양자화로 모든 모델을 동일 조건에서 평가
- **2-stage scoring** — Rule-based 체크 + LLM Judge (Gemini 2.5 Flash) 자동 채점
- **Mean ± Std 리포팅** — 인스턴스 20개의 평균·표준편차로 모델 안정성까지 측정
- **난이도별 분석** — 하/중/상/함정 단위로 rule·judge·combined 점수 분해 저장
- **Full audit trail** — output 전문, judge 프롬프트, raw 응답을 JSON으로 저장 → Gemini 오판 수동 검증 가능
- **Per-model cache** — 모델별 결과 캐시 저장, 재실행 시 완료된 모델 자동 스킵
- **Rich visualization** — 파라미터 크기 vs 성능 산점도(±std 에러바) + 레이더 차트 + 히트맵 다크테마

---

## 📂 File Structure

```
ko-sllm-biz-bench/
├── sllm_eval.py           # 데이터셋 기반 정밀 평가 (메인 평가 스크립트)
├── sllm_benchmark.py      # 단일 프롬프트 벤치마크 (빠른 스팟체크용)
├── eval_cases.json        # 평가 데이터셋 (Task × 20문항, 난이도별)
├── visualize_results.py   # 결과 시각화 (산점도 + 레이더 차트 + 히트맵)
├── pull_models.sh         # Ollama 모델 일괄 다운로드
├── requirements.txt
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Python 3.10+
pip install -r requirements.txt

# Ollama 설치 → https://ollama.com
ollama serve
bash pull_models.sh        # 모델 일괄 다운로드 (~60GB, 시간 걸림)
```

### 2. 환경 변수 설정

```bash
# .env 파일 생성 (또는 export)
GEMINI_API_KEY=your_gemini_api_key_here   # aistudio.google.com/app/apikey
```

### 3. 데이터셋 기반 정밀 평가 (권장)

```bash
# 전체 실행 (14 models × 4 tasks × 20 instances)
python sllm_eval.py

# 모델·Task 필터링 (빠른 파일럿 실행)
python sllm_eval.py --models Qwen2.5-3B Phi-4-Mini Gemma-3-4B
python sllm_eval.py --task task1_json_extraction task4_rag_hallucination

# 캐시 무시하고 전체 재실행
python sllm_eval.py --no-cache
```

### 4. 결과 시각화

```bash
python visualize_results.py               # 기본: .eval_cache 로드 + ≤9B 필터
python visualize_results.py --heatmap     # 히트맵 출력
python visualize_results.py --all-params  # 전체 모델 표시
python visualize_results.py sllm_eval_results_YYYYMMDD_HHMMSS.json  # JSON 직접 지정
```

---

## 📋 Evaluation Tasks & Dataset

평가 데이터셋(`eval_cases.json`)은 한국 실무 환경을 직접 반영해 **한국어로 직접 제작**했습니다.
Task당 20문항 × 4 Task = **총 80문항**, 난이도(하/중/상/함정)별로 구성되어 있습니다.

| Task | 유형 | 채점 방식 | 핵심 역량 | 문항 수 |
|------|------|-----------|-----------|:-------:|
| **Task 1** 회의록 → JSON 추출 | 지시 준수 + 날짜 연산 | Rule + LLM | 포맷 제어, 상대 날짜 계산 | 20 |
| **Task 2** 다중 제약 일정 조율 | 논리 추론 | LLM | 제약 소거, 가능 슬롯 도출 | 20 |
| **Task 3** 이메일 스레드 요약 | 맥락 파악 + 역할 분담 | Rule + LLM | 핵심 결정 추출, Action Item | 20 |
| **Task 4** 사내 규정 기반 답변 | RAG 환각 통제 | Rule + LLM | 규정 외 지어내기 억제 | 20 |

**난이도 분류:**

| 레벨 | 설명 |
|------|------|
| 하/중/상 | 정보 밀도와 제약 조건 복잡도를 단계적으로 높인 문항 |
| 함정 | 모델이 흔히 틀리는 패턴을 의도적으로 설계한 문항 (예: 빈 슬롯 없는 일정, 규정에 없는 내용 질문) |
| 엣지케이스 | 경계 조건 처리 (예: 자정 마감, 중복 일정) |

---

## 🤖 Models (14 models, ≤ 9B)

모두 **Ollama + GGUF 양자화** 기반으로 동일 조건에서 평가합니다.

| Tier | Models |
|------|--------|
| ≤ 2B | Qwen2.5-0.5B, Qwen2.5-1.5B, Gemma-3-1B, Llama-3.2-1B, Gemma-2-2B |
| 3~4B | Qwen2.5-3B, Llama-3.2-3B, Phi-4-Mini (3.8B), Gemma-3-4B |
| 7~9B | Qwen2.5-7B, DeepSeek-R1-7B, EXAONE-3.5-7.8B, Llama-3.1-8B, Gemma-2-9B |

---

## ⚙️ Scoring Pipeline

```
sLLM 응답
   │
   ├── [1단계] Rule-based (40%)
   │     ├─ Task1: JSON 파싱, 날짜 정확도, 포맷 체크
   │     ├─ Task3: 핵심 키워드, Action Item 키워드
   │     └─ Task4: 정답 거절 응답 확인, 환각 키워드 탐지
   │
   └── [2단계] LLM-as-a-Judge — Gemini 2.5 Flash (60%)
         └─ Task별 평가 기준 (1~5점) 자동 채점
              → judge_prompt + judge_raw_response JSON 저장 (수동 검증 가능)

20문항 결과 → mean ± std 집계 (모델 평균 성능 + 안정성 동시 측정)
```

---

## 📊 Output Files

평가 실행 시 타임스탬프 기반 파일 4종이 생성됩니다.

| 파일 | 집계 단위 | 주요 컬럼 |
|------|-----------|-----------|
| `sllm_eval_results_{ts}.csv` | 인스턴스 1건씩 (raw) | instance_id, difficulty, rule_score, judge_score, combined_score |
| `sllm_eval_results_{ts}_summary.csv` | model × task | rule/judge/combined 각각 mean, std, count |
| `sllm_eval_results_{ts}_by_diff.csv` | model × task × difficulty | rule/judge/combined 각각 mean, std, count |
| `sllm_eval_results_{ts}.json` | 인스턴스 1건씩 (전체) | output_full, judge_prompt, judge_raw_response 포함 |
| `*_viz_scatter_radar.png` | — | 산점도(±std 에러바) + 레이더 차트 |

캐시는 `.eval_cache/` 폴더에 모델별로 저장됩니다. `eval_cases.json`이 변경되면 MD5 해시 기반으로 자동 무효화됩니다.

---

## 🛠️ Adding Custom Test Cases

`eval_cases.json`에 인스턴스를 추가하기만 하면 즉시 반영됩니다:

```json
{
  "task1_json_extraction": {
    "instances": [
      {
        "id": "t1_custom_01",
        "difficulty": "중",
        "prompt": "오늘 날짜는 2026년 3월 10일. ...",
        "expected": { "dates": { "홍길동": "2026-03-18" } }
      }
    ]
  }
}
```

---

## 🔒 Security Notes

> ⚠️ **API 키를 절대 코드에 하드코딩하지 마세요.**
> `.env` 파일에 저장하고 `.gitignore`에 추가하세요.

```bash
# .env (절대 커밋 금지)
GEMINI_API_KEY=AIza...
```

---

## 📜 License

MIT
