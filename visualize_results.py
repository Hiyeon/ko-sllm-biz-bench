"""
sLLM 벤치마크 결과 시각화 (산점도 + 레이더 차트 통합, 겹침 방지 적용)
======================================================
실행: python visualize_results.py sllm_benchmark_YYYYMMDD_HHMMSS.json
"""

import json
import sys
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import font_manager
import matplotlib.lines as mlines

# ─────────────────────────────────────────────
# 🎨  폰트 / 테마 스타일 설정
# ─────────────────────────────────────────────

# macOS 한글 폰트 적용 (없으면 시스템 기본 폰트)
KOREAN_FONTS = ["AppleGothic", "NanumGothic", "Malgun Gothic", "DejaVu Sans"]
for font in KOREAN_FONTS:
    if font in [f.name for f in font_manager.fontManager.ttflist]:
        plt.rcParams["font.family"] = font
        break

plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#2e3250",
    "axes.labelcolor":  "#c8cce8",
    "axes.titlecolor":  "#ffffff",
    "text.color":       "#c8cce8",
    "xtick.color":      "#8b8fa8",
    "ytick.color":      "#8b8fa8",
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# ─────────────────────────────────────────────
# 🏷️  모델 계열(Family) 및 체급(Tier) 설정
# ─────────────────────────────────────────────

# 같은 계열의 모델을 동일한 색상/도형으로 묶기 위한 매핑
FAMILY_STYLES = {
    "Qwen":     {"color": "#4fc3f7", "marker": "o"}, # 하늘색, 동그라미
    "Gemma":    {"color": "#81c784", "marker": "s"}, # 연두색, 네모
    "Llama":    {"color": "#ffb74d", "marker": "^"}, # 주황색, 세모(위)
    "Phi":      {"color": "#e57373", "marker": "D"}, # 붉은색, 마름모
    "DeepSeek": {"color": "#ba68c8", "marker": "v"}, # 보라색, 세모(아래)
    "EXAONE":   {"color": "#a1887f", "marker": "*"}, # 갈색, 별
    "Mistral":  {"color": "#f06292", "marker": "p"}, # 핑크색, 오각형
    "기타":     {"color": "#ffffff", "marker": "X"}, # 흰색, X
}

# 겹침 방지를 위해 계열별로 X축(파라미터)에 미세한 오프셋 적용
def get_x_offset(family):
    offsets = {"Qwen": -0.2, "Gemma": 0.0, "Llama": 0.2, "Phi": -0.4, "DeepSeek": 0.4, "EXAONE": -0.1, "Mistral": 0.1}
    return offsets.get(family, 0)

# 배경색 구역 세팅 (전문적인 용어로 변경)
TIER_BGS = [
    (0.0, 2.5,  "#4fc3f7", "초경량 / Nano (≤ 2B)"),
    (2.5, 4.5,  "#81c784", "경량 / Mini (3~4B)"),
    (4.5, 10.0, "#ffb74d", "표준 / Base (7~9B)"),
]

TASK_LABELS = {
    "task1_json_extraction":    "Task 1: JSON 추출 및 포맷팅",
    "task2_scheduling":         "Task 2: 다중 제약 조건 일정 조율",
    "task3_email_summary":      "Task 3: 이메일 맥락 파악 및 요약",
    "task4_rag_hallucination":  "Task 4: RAG 환각(Hallucination) 방어",
}

# ─────────────────────────────────────────────
# 📂  데이터 로드 및 전처리
# ─────────────────────────────────────────────

def load_latest_json(directory: str = ".") -> tuple:
    files = sorted(glob.glob(os.path.join(directory, "sllm_benchmark_*.json")), reverse=True)
    if not files:
        raise FileNotFoundError("sllm_benchmark_*.json 파일이 없습니다.")
    print(f"로드: {files[0]}")
    with open(files[0], encoding="utf-8") as f:
        return json.load(f), files[0]


def load_from_cache(cache_dir: str) -> list:
    """`.eval_cache/` 안의 모든 캐시 JSON을 합쳐 결과 리스트로 반환합니다."""
    files = sorted(glob.glob(os.path.join(cache_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"캐시 파일 없음: {cache_dir}")
    records = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            records.extend(json.load(f))
    models = {r["model"] for r in records}
    print(f"캐시 로드: {len(files)}개 파일, {len(models)}개 모델, 총 {len(records)}건")
    return records

def build_df(records: list) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if "is_error" in df.columns:
        df = df[~df["is_error"]]

    def to_float_param(p):
        try:
            return float(str(p).replace("B", "").replace("b", "").strip())
        except:
            return 0.0
    df["param_val"] = df["params"].apply(to_float_param)

    def to_family(name):
        name = str(name).lower()
        if "qwen" in name: return "Qwen"
        if "gemma" in name: return "Gemma"
        if "llama" in name: return "Llama"
        if "phi" in name: return "Phi"
        if "deepseek" in name: return "DeepSeek"
        if "exaone" in name: return "EXAONE"
        if "mistral" in name or "nemo" in name: return "Mistral"
        return "기타"
    df["family"] = df["model"].apply(to_family)
    
    return df

# ─────────────────────────────────────────────
# 📊  차트 1: 산점도 (글씨 겹침 방지 + std 에러바)
# ─────────────────────────────────────────────

def plot_scatter(ax, df_plot, score_col, title, std_col: str = None):
    """산점도를 그립니다. std_col 이 지정되면 수직 에러바(±std)를 함께 표시합니다."""
    max_param = df_plot["param_val"].max() if not df_plot.empty else 15.0
    x_max = max(max_param * 1.25, max_param + 0.8)
    x_max = max(x_max, 3.0)

    # 1. 체급별 배경색(axvspan) — x_max 범위 안에 걸치는 것만 표시
    for start, end, color, label in TIER_BGS:
        if start >= x_max:
            continue
        clipped_end = min(end, x_max)
        ax.axvspan(start, clipped_end, color=color, alpha=0.04, zorder=0)
        mid_x = (start + clipped_end) / 2
        ax.text(mid_x, 1.05, label, transform=ax.get_xaxis_transform(),
                ha="center", color=color, fontsize=10, alpha=0.8, fontweight="bold")

    points_info = []

    # 2. 산점도(Scatter) + 에러바 그리기 및 좌표 수집
    for family, style in FAMILY_STYLES.items():
        fam_df = df_plot[df_plot["family"] == family]
        if not fam_df.empty:
            for _, row in fam_df.iterrows():
                x_pos = row["param_val"] + get_x_offset(family)
                y_pos = row[score_col]
                yerr  = row[std_col] if (std_col and std_col in row.index and not pd.isna(row[std_col])) else 0

                # 에러바 (std > 0 일 때만 표시) — 얇고 반투명하게
                if yerr > 0:
                    ax.errorbar(x_pos, y_pos, yerr=yerr, fmt="none",
                                ecolor="gray", elinewidth=0.8,
                                alpha=0.5, capsize=2, capthick=0.8, zorder=2)

                ax.scatter(x_pos, y_pos, color=style["color"], marker=style["marker"],
                           s=130, edgecolors="#0f1117", linewidths=1.5, zorder=3)
                points_info.append({"x": x_pos, "y": y_pos, "model": row["model"]})

    # 3. 텍스트 라벨 겹침 방지 — 반복 반발(repulsion) 방식
    MIN_DX, MIN_DY = 0.6, 0.055   # x·y 최소 간격
    ITER = 60                       # 반발 반복 횟수

    label_pos = [[p["x"], p["y"] + 0.04] for p in points_info]

    for _ in range(ITER):
        for i, (lx, ly) in enumerate(label_pos):
            dx_sum, dy_sum = 0.0, 0.0
            for j, (ox, oy) in enumerate(label_pos):
                if i == j:
                    continue
                dx = lx - ox
                dy = ly - oy
                if abs(dx) < MIN_DX and abs(dy) < MIN_DY:
                    push = 0.012
                    dy_sum += push if dy >= 0 else -push
                    dx_sum += (push * 0.3) if dx >= 0 else -(push * 0.3)
            label_pos[i][0] += dx_sum
            label_pos[i][1] = max(0.02, min(1.12, ly + dy_sum))

    for p, (lx, ly) in zip(points_info, label_pos):
        ax.annotate(
            p["model"], xy=(p["x"], p["y"]), xytext=(lx, ly),
            fontsize=8.5, color="#e0e0e0", ha="center", va="center", zorder=4,
            arrowprops=dict(arrowstyle="-", color="#555977", lw=0.7)
                if abs(lx - p["x"]) > 0.15 or abs(ly - p["y"]) > 0.07 else None,
        )

    # 4. 축 및 그리드 설정
    max_param = df_plot["param_val"].max() if not df_plot.empty else 15.0
    x_max = max(max_param * 1.25, max_param + 0.8)   # 여백 확보
    x_max = max(x_max, 3.0)                           # 최소 3B까지는 표시

    ax.set_title(title, fontsize=14, fontweight="bold", pad=28, color="#ffffff")
    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(axis='y', linestyle='--', color="#2e3250", alpha=0.5, zorder=1)
    ax.set_xlabel("파라미터 크기 (Billion)", fontsize=10, color="#8b8fa8")
    ax.set_ylabel("성능 점수 (0~1)", fontsize=10, color="#8b8fa8")
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# ─────────────────────────────────────────────
# 🎯  차트 2: 레이더 차트 (Task 프로필)
# ─────────────────────────────────────────────

def plot_radar(ax, df: pd.DataFrame, top_n: int = 4):
    tasks = [t for t in TASK_LABELS if t in df["task"].unique()]
    n_tasks = len(tasks)
    angles = np.linspace(0, 2 * np.pi, n_tasks, endpoint=False).tolist()
    angles += angles[:1]  # 폐곡선 처리

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    
    # 레이더 차트용 짧은 라벨
    short_labels = ["Task 1\nJSON 추출", "Task 2\n일정 조율", "Task 3\n이메일 요약", "Task 4\nRAG 환각"]
    ax.set_xticklabels(short_labels, fontsize=10, color="#c8cce8", fontweight="bold")
    
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7, color="#666666")
    ax.set_facecolor("#1a1d27")
    ax.grid(color="#2e3250", linewidth=1.0, linestyle="--")
    ax.set_title(f"Task Profile (상위 {top_n}개 모델)", fontsize=14, fontweight="bold", pad=30, color="#ffffff")

    # 상위 모델 선정 (종합 점수 기준)
    top_models = (df.groupby("model")["combined_score"].mean()
                    .sort_values(ascending=False).head(top_n).index.tolist())

    # 레이더 차트 전용 밝고 뚜렷한 색상
    radar_colors = ["#4fc3f7", "#ffb74d", "#81c784", "#ba68c8", "#fff176", "#ff8a65"]

    for i, model in enumerate(top_models):
        vals = []
        for t in tasks:
            row = df[(df["model"] == model) & (df["task"] == t)]
            vals.append(row["combined_score"].mean() if len(row) else 0)
        vals += vals[:1]
        
        color = radar_colors[i % len(radar_colors)]
        ax.plot(angles, vals, "o-", linewidth=2.2, color=color, markersize=5, label=model)
        ax.fill(angles, vals, alpha=0.04, color=color)   # fill 최소화

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              fontsize=9, facecolor="#1a1d27", edgecolor="#2e3250", labelcolor="#c8cce8")

# ─────────────────────────────────────────────
# 🖼️  전체 레이아웃 조합
# ─────────────────────────────────────────────

def render(df: pd.DataFrame, save_path: str):
    # 그리드 3행 4열 (레이더 차트를 우측 상단에 배치하기 위함)
    fig = plt.figure(figsize=(25, 17))
    gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35, top=0.92, bottom=0.06)

    model_count = df["model"].nunique()
    fig.suptitle(f"경량 오픈소스 sLLM 한국어 실무 성능 분석 — ≤ 9B ({model_count}개 모델)",
                 fontsize=19, fontweight="bold", color="white", y=0.98)

    # [1] 상단 좌측 (Row 0, Col 0~2): 종합 평균 점수 산점도
    ax_overall = fig.add_subplot(gs[0, 0:3])
    df_overall = (df.groupby(["model", "param_val", "family"])["combined_score"]
                  .agg(combined_score="mean", combined_score_std="std")
                  .reset_index())
    df_overall["combined_score_std"] = df_overall["combined_score_std"].fillna(0)
    plot_scatter(ax_overall, df_overall, "combined_score", "종합 평균 점수 (Overall Average)", std_col="combined_score_std")

    # 전체 범례 (상단 차트에만 추가)
    handles = []
    for fam, style in FAMILY_STYLES.items():
        if fam in df["family"].values:
            handles.append(mlines.Line2D([], [], color='w', marker=style["marker"], 
                           markerfacecolor=style["color"], markersize=10, label=fam, linestyle=''))
    
    ax_overall.legend(handles=handles, loc='lower right', title="모델 계열 (Family)",
                      facecolor="#1a1d27", edgecolor="#2e3250", fontsize=10, title_fontsize=11)

    # [2] 상단 우측 (Row 0, Col 3): Task 프로필 (레이더 차트)
    ax_radar = fig.add_subplot(gs[0, 3], polar=True)
    plot_radar(ax_radar, df, top_n=6)

    # [3] 하단 (Row 1, Row 2): 4가지 Task 개별 차트
    tasks = [t for t in TASK_LABELS if t in df["task"].unique()]
    for i, task_key in enumerate(tasks):
        row = (i // 2) + 1
        col_start = (i % 2) * 2
        col_end = col_start + 2
        ax = fig.add_subplot(gs[row, col_start:col_end])
        df_task = (df[df["task"] == task_key]
                   .groupby(["model", "param_val", "family"])["combined_score"]
                   .agg(combined_score="mean", combined_score_std="std")
                   .reset_index())
        df_task["combined_score_std"] = df_task["combined_score_std"].fillna(0)
        plot_scatter(ax, df_task, "combined_score", TASK_LABELS[task_key], std_col="combined_score_std")

    # 이미지 저장
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n✅ 시각화 완료! 차트가 저장되었습니다: {save_path}")
    plt.show()

# ─────────────────────────────────────────────
# 🔥  차트 3: 히트맵 (모델 × Task/난이도)
# ─────────────────────────────────────────────

DIFF_ORDER = ["하", "중", "상", "함정", "엣지"]
TASK_SHORT = {
    "task1_json_extraction":   "Task1\nJSON추출",
    "task2_scheduling":        "Task2\n일정조율",
    "task3_email_summary":     "Task3\n이메일요약",
    "task4_rag_hallucination": "Task4\nRAG환각",
}

def render_heatmap(df: pd.DataFrame, save_path: str):
    """모델 × (Task · 난이도) 히트맵을 그립니다."""
    df_valid = df[~df["is_error"]] if "is_error" in df.columns else df

    # ── 피벗: 행=모델, 열=(task, difficulty) ──────────────────────
    agg = (df_valid
           .groupby(["model", "task", "difficulty"])["combined_score"]
           .mean().round(2)
           .reset_index())

    # 열 순서: task 순 × 난이도 순
    task_order = [t for t in TASK_SHORT if t in agg["task"].unique()]
    diff_vals  = [d for d in DIFF_ORDER if d in agg["difficulty"].unique()]
    col_tuples = [(t, d) for t in task_order for d in diff_vals
                  if not agg[(agg["task"] == t) & (agg["difficulty"] == d)].empty]

    pivot = agg.pivot_table(index="model", columns=["task", "difficulty"],
                            values="combined_score", aggfunc="mean")
    pivot = pivot.reindex(columns=pd.MultiIndex.from_tuples(col_tuples), fill_value=np.nan)

    # 행: 종합 평균 내림차순 정렬
    pivot = pivot.reindex(pivot.mean(axis=1).sort_values(ascending=False).index)

    n_rows, n_cols = pivot.shape
    fig, ax = plt.subplots(figsize=(max(14, n_cols * 1.05), max(5, n_rows * 0.7 + 2)))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    data = pivot.values.astype(float)

    # 색상 맵: 0=빨강, 0.5=노랑, 1=초록
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "score", ["#c62828", "#f9a825", "#2e7d32"], N=256)
    cmap.set_bad(color="#2a2d3a")   # NaN = 어두운 회색

    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    # ── 셀 텍스트 ──────────────────────────────────────────────────
    for r in range(n_rows):
        for c in range(n_cols):
            val = data[r, c]
            if np.isnan(val):
                continue
            txt_color = "white" if val < 0.65 else "#0f1117"
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=txt_color, fontweight="bold")

    # ── 축 레이블 ──────────────────────────────────────────────────
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(pivot.index, fontsize=10, color="#c8cce8")

    # Task 구분선 + 열 라벨
    col_labels, task_sep_positions = [], []
    prev_task = None
    for ci, (task, diff) in enumerate(col_tuples):
        col_labels.append(diff)
        if task != prev_task and prev_task is not None:
            task_sep_positions.append(ci - 0.5)
        prev_task = task

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=9, color="#8b8fa8")

    for x in task_sep_positions:
        ax.axvline(x, color="#0f1117", linewidth=2.5, zorder=5)

    # Task 이름을 열 그룹 상단에 표시
    task_positions = {}
    for ci, (task, _) in enumerate(col_tuples):
        task_positions.setdefault(task, []).append(ci)
    for task, cols in task_positions.items():
        mid = np.mean(cols)
        ax.text(mid, -1.6, TASK_SHORT.get(task, task), ha="center", va="bottom",
                fontsize=10, color="#ffffff", fontweight="bold",
                transform=ax.transData)

    # ── 컬러바 ────────────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="#8b8fa8")
    cbar.outline.set_edgecolor("#2e3250")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8b8fa8", fontsize=9)
    cbar.set_label("combined score", color="#8b8fa8", fontsize=10)

    ax.set_title(
        f"모델 × Task/난이도 성능 히트맵  ({n_rows}개 모델)",
        fontsize=15, fontweight="bold", color="white", pad=40)
    ax.tick_params(axis="both", colors="#8b8fa8", length=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2e3250")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"✅ 히트맵 저장: {save_path}")
    plt.show()


# ─────────────────────────────────────────────
# 🚀  실행
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    script_dir = os.path.dirname(os.path.abspath(__file__))

    p = argparse.ArgumentParser(description="sLLM 벤치마크 결과 시각화")
    p.add_argument("json_path", nargs="?", help="결과 JSON 파일 경로 (지정 시 해당 파일 사용)")
    p.add_argument("--from-cache", action="store_true",
                   help=".eval_cache/ 의 모든 캐시 파일을 합쳐서 시각화 (기본 동작)")
    p.add_argument("--max-params", type=float, default=10.0,
                   help="이 값 미만인 모델만 표시 (기본: 10.0 → 9B 이하)")
    p.add_argument("--all-params", action="store_true",
                   help="파라미터 필터 해제 — 전체 모델 표시")
    p.add_argument("--heatmap", action="store_true",
                   help="산점도 대신 모델×Task/난이도 히트맵으로 출력")
    args = p.parse_args()

    # 기본 동작: JSON 미지정이면 캐시에서 로드
    if args.json_path:
        with open(args.json_path, encoding="utf-8") as f:
            records = json.load(f)
        tag = os.path.splitext(os.path.basename(args.json_path))[0]
    else:
        cache_dir = os.path.join(script_dir, ".eval_cache")
        records   = load_from_cache(cache_dir)
        tag       = "cache"

    df = build_df(records)

    if not args.all_params:
        before = df["model"].nunique()
        df = df[df["param_val"] < args.max_params]
        after = df["model"].nunique()
        print(f"파라미터 필터 (<{args.max_params}B): {before}개 → {after}개 모델")
        tag += f"_under{args.max_params}B"

    if df.empty:
        print("결과 데이터가 없습니다.")
        sys.exit(1)

    if args.heatmap:
        out_path = os.path.join(script_dir, f"{tag}_viz_heatmap.png")
        render_heatmap(df, save_path=out_path)
    else:
        out_path = os.path.join(script_dir, f"{tag}_viz_scatter_radar.png")
        render(df, save_path=out_path)
