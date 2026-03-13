"""
오픈소스 sLLM 실무 성능 벤치마크 — 데이터셋 기반 정밀 평가
============================================================
eval_cases.json 에 정의된 Task당 20개 문항을 로드하여
모델별 평균 점수를 산출합니다.

[실행 예시]
  python sllm_eval.py
  python sllm_eval.py --backend ollama
  python sllm_eval.py --backend huggingface
  python sllm_eval.py --dataset eval_cases.json --models Qwen2.5-3B Phi-4-Mini
  python sllm_eval.py --task task1_json_extraction task4_rag_hallucination

[환경 변수]
  GEMINI_API_KEY  Gemini Judge API 키 (필수)
  HF_TOKEN        Llama·Gemma 등 gated 모델용 HuggingFace 토큰
"""

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
import warnings
from datetime import datetime

import requests
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
os.environ.setdefault("HF_HUB_VERBOSITY", "warning")

import pandas as pd
import torch
from google import genai
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# ─────────────────────────────────────────────
# ⚙️  설정 (Settings)
# ─────────────────────────────────────────────

# 환경 변수에서 읽기 (.env 파일 또는 export GEMINI_API_KEY=...)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
HF_TOKEN       = os.environ.get("HF_TOKEN", "")
OLLAMA_BASE_URL = "http://localhost:11434"

# 기본 데이터셋 파일 경로 (스크립트와 동일 폴더)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(_SCRIPT_DIR, "eval_cases.json")
_CACHE_DIR = os.path.join(_SCRIPT_DIR, ".eval_cache")

# ── 모델 목록 ────────────────────────────────
MODELS = [
    # ≤ 2B
    {"name": "Qwen2.5-0.5B",     "ollama": "qwen2.5:0.5b",      "hf": "Qwen/Qwen2.5-0.5B-Instruct",                      "gated": False, "params": "0.5B"},
    {"name": "Gemma-3-1B",       "ollama": "gemma3:1b",          "hf": "google/gemma-3-1b-it",                             "gated": True,  "params": "1B"},
    {"name": "Llama-3.2-1B",     "ollama": "llama3.2:1b",        "hf": "meta-llama/Llama-3.2-1B-Instruct",                 "gated": True,  "params": "1B"},
    {"name": "Qwen2.5-1.5B",     "ollama": "qwen2.5:1.5b",       "hf": "Qwen/Qwen2.5-1.5B-Instruct",                      "gated": False, "params": "1.5B"},
    {"name": "Gemma-2-2B",       "ollama": "gemma2:2b",          "hf": "google/gemma-2-2b-it",                             "gated": True,  "params": "2B"},
    # 3~4B
    {"name": "Llama-3.2-3B",     "ollama": "llama3.2:3b",        "hf": "meta-llama/Llama-3.2-3B-Instruct",                 "gated": True,  "params": "3B"},
    {"name": "Qwen2.5-3B",       "ollama": "qwen2.5:3b",         "hf": "Qwen/Qwen2.5-3B-Instruct",                        "gated": False, "params": "3B"},
    {"name": "Phi-4-Mini",       "ollama": "phi4-mini",          "hf": "microsoft/Phi-4-mini-instruct",                    "gated": False, "params": "3.8B"},
    {"name": "Gemma-3-4B",       "ollama": "gemma3:4b",          "hf": "google/gemma-3-4b-it",                             "gated": True,  "params": "4B"},
    # 7~8B
    {"name": "Qwen2.5-7B",       "ollama": "qwen2.5:7b",         "hf": "Qwen/Qwen2.5-7B-Instruct",                        "gated": False, "params": "7B"},
    {"name": "DeepSeek-R1-7B",   "ollama": "deepseek-r1:7b",     "hf": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",          "gated": False, "params": "7B"},
    {"name": "EXAONE-3.5-7.8B",  "ollama": "exaone3.5:7.8b",     "hf": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",            "gated": False, "params": "7.8B"},
    {"name": "Llama-3.1-8B",     "ollama": "llama3.1:8b",        "hf": "meta-llama/Meta-Llama-3.1-8B-Instruct",            "gated": True,  "params": "8B"},
    # 9B
    {"name": "Gemma-2-9B",       "ollama": "gemma2:9b",          "hf": "google/gemma-2-9b-it",                             "gated": True,  "params": "9B"},
]

console = Console()


# ─────────────────────────────────────────────
# 📂  데이터셋 로더
# ─────────────────────────────────────────────

def load_dataset(filepath: str) -> dict:
    """JSON 파일에서 평가 데이터셋을 로드합니다."""
    if not os.path.exists(filepath):
        # 스크립트 디렉터리 기준 재시도
        alt = os.path.join(_SCRIPT_DIR, filepath)
        if os.path.exists(alt):
            filepath = alt
        else:
            console.print(f"[red]❌ '{filepath}' 파일을 찾을 수 없습니다.[/red]")
            sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        try:
            dataset = json.load(f)
        except json.JSONDecodeError as e:
            console.print(f"[red]❌ JSON 파싱 에러: {e}[/red]")
            sys.exit(1)

    total = sum(len(v["instances"]) for v in dataset.values())
    console.print(f"[green]✅ 데이터셋 로드: {len(dataset)}개 Task, 총 {total}개 문항[/green]")
    return dataset


# ─────────────────────────────────────────────
# 💾  모델별 평가 결과 캐시
# ─────────────────────────────────────────────

def _dataset_hash(dataset: dict) -> str:
    """데이터셋 내용 기반 8자리 해시 — eval_cases.json 변경 시 캐시 자동 무효화."""
    content = json.dumps(dataset, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()[:8]


def _cache_path(model_name: str, d_hash: str, task_keys: list[str]) -> str:
    safe  = re.sub(r"[^\w\-]", "_", model_name)
    tasks = hashlib.md5("".join(sorted(task_keys)).encode()).hexdigest()[:6]
    return os.path.join(_CACHE_DIR, f"{safe}__{d_hash}__{tasks}.json")


def load_model_cache(model_name: str, dataset: dict, task_keys: list[str]) -> list | None:
    """캐시 파일이 있으면 결과 list를 반환, 없으면 None."""
    path = _cache_path(model_name, _dataset_hash(dataset), task_keys)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_model_cache(model_name: str, dataset: dict, task_keys: list[str], model_results: list):
    """모델 평가 완료 직후 결과를 캐시 파일로 저장."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = _cache_path(model_name, _dataset_hash(dataset), task_keys)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model_results, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────
# 🤖  백엔드별 모델 호출
# ─────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available():   return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


# ── Ollama ──

def call_ollama(model_id: str, prompt: str, timeout: int = 180) -> tuple[str, float]:
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model_id, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.1, "num_predict": 1024},
    }
    start = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "").strip(), round(time.time() - start, 2)
    except requests.exceptions.ConnectionError:
        return "[ERROR] Ollama 서버 연결 실패. `ollama serve` 실행 여부를 확인하세요.", 0.0
    except requests.exceptions.Timeout:
        return f"[ERROR] {timeout}초 내 응답 없음 (타임아웃)", 0.0
    except Exception as e:
        return f"[ERROR] {e}", 0.0


def check_ollama(model_id: str) -> bool:
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        installed = [m["name"] for m in resp.json().get("models", [])]
        if model_id in installed or f"{model_id}:latest" in installed:
            return True
        console.print(f"  [yellow]⚠️ Ollama에 '{model_id}' 없음 → ollama pull {model_id}[/yellow]")
        return False
    except requests.exceptions.ConnectionError:
        console.print("  [red]❌ Ollama 서버 연결 불가. `ollama serve`를 먼저 실행하세요.[/red]")
        return False
    except Exception as e:
        console.print(f"  [yellow]⚠️ Ollama 확인 실패: {e}[/yellow]")
        return False


# ── HuggingFace ──

def _build_chat_prompt(tokenizer, user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return user_message


def call_hf_model(model_id: str, prompt: str, max_new_tokens: int = 1024) -> tuple[str, float]:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = get_device()
    token = HF_TOKEN or None
    start = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id, token=token, dtype=dtype,
            device_map="auto" if device == "cuda" else None, trust_remote_code=True,
        )
        if device != "cuda":
            model = model.to(device)
        model.eval()
        inputs = tokenizer(_build_chat_prompt(tokenizer, prompt), return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=1.0, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip(), round(time.time() - start, 2)
    except Exception as e:
        return f"[ERROR] {model_id}: {e}", 0.0
    finally:
        try:
            del model
        except NameError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def check_hf_model(model_id: str, gated: bool) -> bool:
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=model_id, filename="config.json", token=HF_TOKEN or None)
        return True
    except Exception as e:
        err = str(e)
        if "401" in err or "gated" in err.lower() or "restricted" in err.lower():
            console.print(f"  [red]🔒 게이티드 모델 — HF 토큰 및 사용 동의 필요[/red]")
            console.print(f"     → https://huggingface.co/{model_id}")
        else:
            console.print(f"  [yellow]⚠️ 모델 접근 불가: {err[:120]}[/yellow]")
        return False


# ── 통합 디스패처 ──

def call_model(model_info: dict, prompt: str, backend: str) -> tuple[str, float]:
    if backend == "ollama":
        return call_ollama(model_info["ollama"], prompt)
    else:
        return call_hf_model(model_info["hf"], prompt)


def check_model(model_info: dict, backend: str) -> bool:
    if backend == "ollama":
        return check_ollama(model_info["ollama"])
    else:
        return check_hf_model(model_info["hf"], model_info.get("gated", False))


# ─────────────────────────────────────────────
# ✅  규칙 기반 평가
# ─────────────────────────────────────────────

def evaluate_task1_rules(output: str, expected: dict) -> dict:
    scores, details = {}, []

    # JSON 추출 (마크다운 코드블록 제거)
    cleaned = re.sub(r"```(?:json)?|```", "", output).strip()
    match = re.search(r"(\[.*\]|\{.*\})", cleaned, re.DOTALL)
    cleaned = match.group(1).strip() if match else cleaned

    try:
        json_data = json.loads(cleaned)
        scores["json_parseable"] = 1.0
        details.append("✅ JSON 파싱 성공")
    except json.JSONDecodeError:
        return {"scores": {"json_parseable": 0.0}, "details": ["❌ JSON 파싱 실패"], "total": 0.0}

    # 날짜 정확도 (Ground Truth 기반)
    expected_dates = expected.get("dates", {})
    if expected_dates:
        output_text = json.dumps(json_data, ensure_ascii=False)
        hits = sum(1 for d in expected_dates.values() if d in output_text)
        scores["date_accuracy"] = round(hits / len(expected_dates), 2)
        details.append(f"날짜 정확도: {hits}/{len(expected_dates)}")

    # 포맷 통제 (사족 없는지)
    noise = any(re.search(p, output) for p in [r"결과는 다음과 같", r"안녕하세요", r"물론입니다", r"아래와 같이"])
    scores["format_clean"] = 0.0 if noise else 1.0
    details.append("❌ 사족 포함" if noise else "✅ 포맷 깔끔")

    return {"scores": scores, "details": details, "total": round(sum(scores.values()) / len(scores), 2)}


def evaluate_task3_rules(output: str, expected: dict) -> dict:
    scores, details = {}, []

    venue = expected.get("final_venue", "")
    if venue:
        scores["final_venue"] = 1.0 if venue in output else 0.0
        details.append(f"{'✅' if scores['final_venue'] else '❌'} 장소: {venue}")

    action_items = expected.get("action_items", {})
    if action_items:
        hits = sum(1 for p, k in action_items.items() if p in output and k in output)
        scores["action_items"] = round(hits / len(action_items), 2)
        details.append(f"Action Item: {hits}/{len(action_items)}")

    return {"scores": scores, "details": details, "total": round(sum(scores.values()) / len(scores), 2) if scores else 0.0}


def evaluate_task4_rules(output: str, expected: dict) -> dict:
    scores, details = {}, []

    correct = expected.get("correct_answer", "알 수 없")
    if correct in output:
        scores["correct_refusal"] = 1.0
        details.append("✅ 정답 거절 응답 확인")
    else:
        scores["correct_refusal"] = 0.0
        details.append("❌ 정답 거절 없음")

    hall = [kw for kw in expected.get("hallucination_keywords", []) if kw in output]
    scores["no_hallucination"] = 0.0 if hall else 1.0
    details.append(f"❌ 환각 감지: {hall}" if hall else "✅ 환각 없음")

    return {"scores": scores, "details": details, "total": round(sum(scores.values()) / len(scores), 2)}


JUDGE_SCHEMA = {
    "task1_json_extraction":   '{"date_accuracy": <1-5>, "format_compliance": <1-5>, "completeness": <1-5>, "reason": "<이유>"}',
    "task2_scheduling":        '{"constraint_coverage": <1-5>, "correct_slot": <1-5>, "reasoning_quality": <1-5>, "reason": "<이유>"}',
    "task3_email_summary":     '{"final_decision_accuracy": <1-5>, "action_item_accuracy": <1-5>, "no_hallucination": <1-5>, "reason": "<이유>"}',
    "task4_rag_hallucination": '{"hallucination_control": <1-5>, "refusal_quality": <1-5>, "boundary_respect": <1-5>, "reason": "<이유>"}',
}


# ─────────────────────────────────────────────
# 🧑‍⚖️  LLM-as-a-Judge (Gemini)
# ─────────────────────────────────────────────

def llm_judge(judge_client, task_key: str, prompt: str, expected: dict, output: str) -> dict:
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY 없음", "scores": {}, "total": 0.0}

    expected_str = json.dumps(expected, ensure_ascii=False, indent=2)
    schema = JUDGE_SCHEMA.get(task_key, '{"quality": <1-5>, "reason": "<이유>"}')

    judge_prompt = f"""너는 사내 AI 시스템을 평가하는 깐깐한 심판관이야.
다음은 모델에게 주어진 [사용자 입력]과, 기획자가 설정한 [기대 정답(Ground Truth)]이야.

[사용자 입력]:
{prompt}

[기대 정답 (Ground Truth)]:
{expected_str}

[sLLM 응답]:
{output}

위 기댓값을 바탕으로 sLLM 응답을 평가해 줘. 각 항목은 1~5점으로 매겨.
결과는 반드시 아래 JSON 형식으로만 출력해. 부연 설명 금지.

{schema}"""

    def _parse_judge_json(raw: str) -> dict | None:
        """Gemini 응답에서 JSON을 추출합니다. 파싱 실패 시 여러 방법을 시도합니다."""
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        candidate = m.group()

        # 1차 시도: 그대로 파싱
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # 2차 시도: reason 필드를 제거하고 숫자 점수만 regex로 추출
        numeric = {}
        for k, v in re.findall(r'"(\w+)"\s*:\s*([1-5](?:\.\d+)?)', candidate):
            if k != "reason":
                numeric[k] = float(v)
        reason_m = re.search(r'"reason"\s*:\s*"(.*?)"(?:\s*[,}])', candidate, re.DOTALL)
        reason = reason_m.group(1) if reason_m else ""

        if numeric:
            return {**numeric, "reason": reason}
        return None

    last_err = ""
    for attempt in range(3):
        try:
            response = judge_client.models.generate_content(model="gemini-2.5-flash", contents=judge_prompt)
            raw = response.text.strip()
            data = _parse_judge_json(raw)
            if data:
                reason = data.pop("reason", "")
                numeric = {k: v for k, v in data.items() if isinstance(v, (int, float))}
                avg = round(sum(numeric.values()) / len(numeric) / 5.0, 2) if numeric else 0.0
                return {"scores": numeric, "reason": reason, "total": avg,
                        "judge_prompt": judge_prompt, "judge_raw_response": raw}
            return {"error": f"JSON 파싱 실패: {raw[:80]}", "scores": {}, "total": 0.0,
                    "judge_prompt": judge_prompt, "judge_raw_response": raw}
        except Exception as e:
            last_err = str(e)
            if "503" in last_err and attempt < 2:
                wait = 8 * (attempt + 1)   # 8s → 16s
                console.print(f"    [dim]⏳ Gemini 503 — {wait}초 후 재시도 ({attempt + 1}/2)...[/dim]")
                time.sleep(wait)
                continue
            break
    return {"error": last_err, "scores": {}, "total": 0.0, "judge_prompt": judge_prompt, "judge_raw_response": ""}


# ─────────────────────────────────────────────
# 🏃  벤치마크 실행 루프
# ─────────────────────────────────────────────

def run_eval(models: list, dataset: dict, backend: str, use_cache: bool = True) -> list[dict]:
    # Gemini Judge 초기화
    judge_client = None
    if GEMINI_API_KEY:
        try:
            judge_client = genai.Client(api_key=GEMINI_API_KEY)
            console.print("[green]✅ Gemini Judge 초기화 완료[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Gemini 초기화 실패: {e} → LLM Judge 건너뜀[/yellow]")
    else:
        console.print("[yellow]⚠️ GEMINI_API_KEY 미설정 → LLM Judge 건너뜀 (규칙 기반만 실행)[/yellow]")

    device_label = "Ollama 서버" if backend == "ollama" else get_device().upper()
    console.print(f"[dim]🖥️  백엔드: {'Ollama' if backend == 'ollama' else 'HuggingFace'} ({device_label})[/dim]")
    if use_cache:
        console.print(f"[dim]💾  캐시 디렉터리: {_CACHE_DIR}[/dim]")
    console.print()

    task_keys = list(dataset.keys())
    results = []

    for model_info in models:
        tag = f"ollama: {model_info['ollama']}"
        console.rule(f"[bold cyan]{model_info['name']}  [{tag}][/bold cyan]")

        # ── 캐시 확인 ──
        if use_cache:
            cached = load_model_cache(model_info["name"], dataset, task_keys)
            if cached is not None:
                console.print(f"  [dim]✅ 캐시 로드 — 추론 건너뜀 ({len(cached)}건)[/dim]")
                results.extend(cached)
                continue

        if not check_model(model_info, backend):
            console.print("[red]❌ 건너뜁니다.[/red]")
            continue

        model_skip = False
        model_results: list[dict] = []

        for task_key, task_data in dataset.items():
            console.print(f"\n  [bold]{task_data['name']}[/bold]")

            task_scores = []

            for idx, instance in enumerate(task_data["instances"]):
                inst_id = instance.get("id", f"inst_{idx}")
                diff    = instance.get("difficulty", "중")

                if model_skip:
                    console.print(f"    [{inst_id}] [dim]⏭️ 접근 오류로 건너뜀[/dim]")
                    continue

                with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True, console=console) as p:
                    p.add_task(f"    [{inst_id} / {diff}] 추론 중...", total=None)
                    output, elapsed = call_model(model_info, instance["prompt"], backend)

                is_error = output.startswith("[ERROR]")
                if is_error:
                    console.print(f"    [{inst_id}] [red]{output[:160]}[/red]")
                    if "401" in output or "gated" in output.lower():
                        model_skip = True
                    combined = 0.0
                    rule_res = {"scores": {}, "details": [], "total": 0.0}
                    judge_res = {"scores": {}, "total": 0.0}
                else:
                    # 규칙 기반 평가
                    rule_res = {"scores": {}, "details": [], "total": 0.0}
                    eval_type = task_data.get("eval_type", "llm")
                    if eval_type in ("rule", "rule+llm"):
                        if "task1" in task_key:
                            rule_res = evaluate_task1_rules(output, instance["expected"])
                        elif "task3" in task_key:
                            rule_res = evaluate_task3_rules(output, instance["expected"])
                        elif "task4" in task_key:
                            rule_res = evaluate_task4_rules(output, instance["expected"])

                    # LLM Judge
                    judge_res = {"scores": {}, "total": 0.0}
                    if judge_client and eval_type in ("llm", "rule+llm"):
                        with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True, console=console) as p:
                            p.add_task("    Gemini 채점 중...", total=None)
                            judge_res = llm_judge(judge_client, task_key, instance["prompt"],
                                                  instance["expected"], output)
                        if "error" in judge_res:
                            console.print(f"    [yellow]⚠️ Judge 오류: {judge_res['error']}[/yellow]")

                    # 종합 점수
                    if rule_res["total"] > 0 and judge_res.get("total", 0) > 0:
                        combined = round(rule_res["total"] * 0.4 + judge_res["total"] * 0.6, 2)
                    elif rule_res["total"] > 0:
                        combined = rule_res["total"]
                    elif judge_res.get("total", 0) > 0:
                        combined = judge_res["total"]
                    else:
                        combined = 0.0

                task_scores.append(combined)
                reason_snippet = judge_res.get("reason", "")[:60]
                console.print(f"    [{inst_id}/{diff}] 점수: [bold]{combined:.2f}[/bold] ({elapsed}s) {reason_snippet}")

                model_results.append({
                    "model":            model_info["name"],
                    "params":           model_info.get("params", "?"),
                    "backend":          backend,
                    "task":             task_key,
                    "task_name":        task_data["name"],
                    "instance_id":      inst_id,
                    "difficulty":       diff,
                    "elapsed_sec":      elapsed,
                    "is_error":         is_error,
                    "rule_score":       rule_res["total"],
                    "judge_score":      judge_res.get("total", 0.0),
                    "combined_score":   combined,
                    "output_full":      output,
                    "output_preview":   output[:120].replace("\n", " ") + ("..." if len(output) > 120 else ""),
                    "rule_details":     " | ".join(rule_res.get("details", [])),
                    "judge_scores":     judge_res.get("scores", {}),
                    "judge_reason":     judge_res.get("reason", ""),
                    "judge_prompt":     judge_res.get("judge_prompt", ""),
                    "judge_raw_response": judge_res.get("judge_raw_response", ""),
                })

            if task_scores:
                avg = round(sum(task_scores) / len(task_scores), 2)
                console.print(f"  [cyan]→ {task_data['name']} 평균: {avg:.2f} ({len(task_scores)}문항)[/cyan]")

        # ── 모델 완료 → 캐시 저장 ──
        if use_cache and model_results:
            save_model_cache(model_info["name"], dataset, task_keys, model_results)
            console.print(f"  [dim]💾 캐시 저장: {model_info['name']}[/dim]")

        results.extend(model_results)

    return results


# ─────────────────────────────────────────────
# 📊  결과 출력 및 저장
# ─────────────────────────────────────────────

def print_summary_table(results: list[dict]):
    if not results:
        console.print("[red]결과가 없습니다.[/red]")
        return

    df = pd.DataFrame(results)
    df_valid = df[~df["is_error"]]

    mean_pv = (df_valid
               .pivot_table(index="model", columns="task", values="combined_score", aggfunc="mean")
               .round(3))
    std_pv  = (df_valid
               .pivot_table(index="model", columns="task", values="combined_score", aggfunc="std")
               .fillna(0).round(3))

    mean_pv["총평균"] = mean_pv.mean(axis=1).round(3)
    std_pv["총평균"]  = std_pv.mean(axis=1).round(3)
    mean_pv = mean_pv.sort_values("총평균", ascending=False)
    std_pv  = std_pv.reindex(mean_pv.index)

    if "params" in df.columns:
        pm = df.drop_duplicates("model").set_index("model")["params"].to_dict()
        new_idx = [f"{m} ({pm.get(m, '?')})" for m in mean_pv.index]
        mean_pv.index = new_idx
        std_pv.index  = new_idx

    table = Table(title="📊 eval_cases 벤치마크 결과 (mean ± std, 0~1)", show_lines=True)
    table.add_column("모델 (파라미터)", style="bold cyan", no_wrap=True)
    for col in mean_pv.columns:
        table.add_column(str(col), justify="center")

    for model_name in mean_pv.index:
        cells = []
        for col in mean_pv.columns:
            m = mean_pv.loc[model_name, col]
            s = std_pv.loc[model_name, col]
            if pd.isna(m):
                cells.append("[dim]-[/dim]")
            else:
                label = f"{m:.2f}[dim]±{s:.2f}[/dim]"
                if m >= 0.8:
                    cells.append(f"[green]{label}[/green]")
                elif m >= 0.6:
                    cells.append(f"[yellow]{label}[/yellow]")
                else:
                    cells.append(f"[red]{label}[/red]")
        table.add_row(str(model_name), *cells)

    console.print(table)


def _agg_scores(grp_df: "pd.DataFrame") -> "pd.DataFrame":
    """rule_score / judge_score / combined_score 각각 mean, std, count 집계."""
    rows = []
    for score_col in ("rule_score", "judge_score", "combined_score"):
        if score_col not in grp_df.columns:
            continue
        agg = (grp_df[score_col]
               .agg(["mean", "std", "count"])
               .rename({"mean": f"{score_col}_mean",
                        "std":  f"{score_col}_std",
                        "count": f"{score_col}_count"}))
        rows.append(agg)
    return pd.concat(rows) if rows else pd.Series(dtype=float)


def save_results(results: list[dict]) -> tuple[str, str]:
    """스크립트 폴더에 3종 CSV + JSON(전체 로그)으로 저장합니다.

    ① sllm_eval_results_{ts}.csv           — 인스턴스별 상세 (raw)
    ② sllm_eval_results_{ts}_summary.csv   — model × task 평균±std
    ③ sllm_eval_results_{ts}_by_diff.csv   — model × task × difficulty 평균±std
       (rule_score / judge_score / combined_score 각각)
    ④ sllm_eval_results_{ts}.json          — 전체 로그 (judge 응답 포함)
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_cols = ["model", "params", "backend", "task", "task_name", "instance_id", "difficulty",
                "elapsed_sec", "rule_score", "judge_score", "combined_score",
                "output_preview", "rule_details", "judge_reason", "is_error"]
    csv_path     = os.path.join(_SCRIPT_DIR, f"sllm_eval_results_{ts}.csv")
    summary_path = os.path.join(_SCRIPT_DIR, f"sllm_eval_results_{ts}_summary.csv")
    diff_path    = os.path.join(_SCRIPT_DIR, f"sllm_eval_results_{ts}_by_diff.csv")
    json_path    = os.path.join(_SCRIPT_DIR, f"sllm_eval_results_{ts}.json")

    df = pd.DataFrame(results)
    df_valid = df[~df["is_error"]]

    # ① 인스턴스별 상세 CSV
    df[[c for c in csv_cols if c in df.columns]].to_csv(csv_path, index=False, encoding="utf-8-sig")
    console.print(f"\n[green]💾 CSV  (인스턴스별 상세): {csv_path}[/green]")

    if not df_valid.empty:
        # ② model × task  mean/std 요약 (combined_score 기준)
        summary_rows = []
        score_cols = [c for c in ("rule_score", "judge_score", "combined_score") if c in df_valid.columns]
        for grp_keys, grp_df in df_valid.groupby(["model", "params", "task", "task_name"]):
            row = dict(zip(["model", "params", "task", "task_name"], grp_keys))
            for sc in score_cols:
                row[f"{sc}_mean"]  = round(grp_df[sc].mean(), 3)
                row[f"{sc}_std"]   = round(grp_df[sc].std(ddof=1) if len(grp_df) > 1 else 0.0, 3)
                row[f"{sc}_count"] = int(grp_df[sc].count())
            summary_rows.append(row)
        # 총평균 행
        for grp_keys, grp_df in df_valid.groupby(["model", "params"]):
            row = {"model": grp_keys[0], "params": grp_keys[1], "task": "ALL", "task_name": "총평균"}
            for sc in score_cols:
                row[f"{sc}_mean"]  = round(grp_df[sc].mean(), 3)
                row[f"{sc}_std"]   = round(grp_df[sc].std(ddof=1) if len(grp_df) > 1 else 0.0, 3)
                row[f"{sc}_count"] = int(grp_df[sc].count())
            summary_rows.append(row)

        pd.DataFrame(summary_rows).sort_values(["model", "task"]).to_csv(
            summary_path, index=False, encoding="utf-8-sig")
        console.print(f"[green]💾 CSV  (model×task mean/std 요약): {summary_path}[/green]")

        # ③ model × task × difficulty  (rule_score / judge_score / combined_score 각각)
        diff_rows = []
        DIFF_ORDER = ["하", "중", "상", "함정", "엣지"]   # 정렬용
        for grp_keys, grp_df in df_valid.groupby(["model", "params", "task", "task_name", "difficulty"]):
            row = dict(zip(["model", "params", "task", "task_name", "difficulty"], grp_keys))
            for sc in score_cols:
                row[f"{sc}_mean"]  = round(grp_df[sc].mean(), 3)
                row[f"{sc}_std"]   = round(grp_df[sc].std(ddof=1) if len(grp_df) > 1 else 0.0, 3)
                row[f"{sc}_count"] = int(grp_df[sc].count())
            diff_rows.append(row)

        diff_df = pd.DataFrame(diff_rows)
        diff_df["_diff_order"] = diff_df["difficulty"].apply(
            lambda x: DIFF_ORDER.index(x) if x in DIFF_ORDER else 99)
        diff_df.sort_values(["model", "task", "_diff_order"]).drop(columns="_diff_order").to_csv(
            diff_path, index=False, encoding="utf-8-sig")
        console.print(f"[green]💾 CSV  (model×task×difficulty mean/std): {diff_path}[/green]")

    # ④ 전체 로그 JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    console.print(f"[green]💾 JSON (전체 로그): {json_path}[/green]")
    console.print("[dim]   → output_full / judge_prompt / judge_raw_response 포함[/dim]")

    return csv_path, json_path


# ─────────────────────────────────────────────
# 🚀  CLI 파싱 및 메인 실행
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="sLLM 데이터셋 기반 평가")
    p.add_argument("--dataset",   default=DEFAULT_DATASET, help="평가 데이터셋 JSON 경로")
    p.add_argument("--backend",   default="ollama", choices=["ollama", "huggingface"], help="추론 백엔드")
    p.add_argument("--models",    nargs="+", help="평가할 모델 name 목록 (미지정 시 전체)")
    p.add_argument("--task",      nargs="+", dest="tasks", help="평가할 task key 목록 (미지정 시 전체)")
    p.add_argument("--no-cache",  action="store_true", help="캐시 무시하고 전체 재실행")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    console.print(Panel.fit(
        "[bold]오픈소스 sLLM 데이터셋 기반 정밀 평가[/bold]\n"
        f"백엔드: {args.backend.upper()} | 데이터셋: {os.path.basename(args.dataset)}",
        border_style="cyan",
    ))

    # 데이터셋 로드
    dataset = load_dataset(args.dataset)

    # Task 필터링
    if args.tasks:
        dataset = {k: v for k, v in dataset.items() if k in args.tasks}
        if not dataset:
            console.print(f"[red]❌ 해당 task가 없습니다: {args.tasks}[/red]")
            sys.exit(1)

    # 모델 필터링
    models = MODELS
    if args.models:
        models = [m for m in MODELS if m["name"] in args.models]
        if not models:
            console.print(f"[red]❌ 해당 모델이 없습니다: {args.models}[/red]")
            console.print(f"[dim]사용 가능: {[m['name'] for m in MODELS]}[/dim]")
            sys.exit(1)

    # 평가 실행
    results = run_eval(models=models, dataset=dataset, backend=args.backend,
                       use_cache=not args.no_cache)

    if results:
        console.print("\n")
        print_summary_table(results)
        save_results(results)

        df = pd.DataFrame(results)
        valid = df[~df["is_error"]]
        if not valid.empty:
            best  = valid.groupby("model")["combined_score"].mean().idxmax()
            bscore = valid.groupby("model")["combined_score"].mean().max()
            console.print(Panel(
                f"[bold green]🏆 종합 최우수 모델: {best}[/bold green]\n"
                f"평균 종합 점수: {bscore:.2f} / 1.00",
                border_style="green",
            ))
    else:
        console.print("[red]\n실행 가능한 결과가 없습니다.[/red]")
