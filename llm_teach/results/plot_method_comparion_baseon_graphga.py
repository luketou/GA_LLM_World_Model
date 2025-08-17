#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.switch_backend("Agg")

# ---------- I/O ----------
def load_scores(path: str, tag: str):
    if not os.path.isfile(path):
        print(f"[WARN] missing file for {tag}: {path} -> skipped")
        return None
    df = pd.read_csv(path)
    need = {"generation", "smiles", "score"}
    if not need.issubset(df.columns):
        raise ValueError(f"{tag} must contain columns {need}, got {list(df.columns)}")
    df = df[["generation", "score"]].copy()
    df["method"] = tag
    return df

# ---------- per-generation stats ----------
def per_gen_stats(df: pd.DataFrame):
    g = df.groupby("generation")
    stats = g["score"].agg(n="count", max="max", avg="mean", min="min").reset_index()
    sorted_arrays = g["score"].apply(lambda s: np.sort(s.to_numpy())[::-1])
    return stats, sorted_arrays

def top_group_curves(sorted_arrays, top_k_max=10):
    gens = sorted(sorted_arrays.index.tolist())
    top1, top2_5, top6_10 = [], [], []
    for g in gens:
        arr = sorted_arrays.loc[g]
        t1 = arr[0] if arr.size >= 1 else np.nan
        hi = min(arr.size, 5)
        t25 = np.nan if hi <= 1 else float(np.mean(arr[1:hi]))
        hi2 = min(arr.size, top_k_max)
        t6_10 = np.nan if hi2 <= 5 else float(np.mean(arr[5:hi2]))
        top1.append(t1); top2_5.append(t25); top6_10.append(t6_10)
    return np.array(gens), np.array(top1), np.array(top2_5), np.array(top6_10)

def winrate_vs_base(base_series: pd.Series, method_series: pd.Series):
    x = base_series.align(method_series, join="inner")
    base = x[0]; meth = x[1]
    mask = base.notna() & meth.notna()
    base = base[mask]; meth = meth[mask]
    if len(base) == 0:
        return 0.0, 0, 0
    wins = (meth > base).sum()
    total = len(base)
    return wins / total, wins, total

# ---------- plotting helpers ----------
def plot_comparison(stats_map, out_png):
    colors = {
        "GraphGA": ("tab:blue", "tab:cyan", "tab:purple"),
        "LLM": ("tab:red", "orange", "firebrick"),
        "LLM_alt": ("tab:green", "limegreen", "seagreen"),
    }
    plt.figure(figsize=(12, 6))
    for name in ["GraphGA", "LLM", "LLM_alt"]:
        if name not in stats_map: 
            continue
        s = stats_map[name].sort_values("generation")
        g = s["generation"].to_numpy()
        cmax, cavg, cmin = colors[name]
        plt.plot(g, s["max"], label=f"{name} Max", linewidth=1.6, color=cmax)
        plt.plot(g, s["avg"], label=f"{name} Avg", linewidth=1.3, color=cavg)
        plt.plot(g, s["min"], label=f"{name} Min", linewidth=1.0, color=cmin, alpha=0.9)
    plt.xlabel("Generation"); plt.ylabel("Score")
    plt.title("Max / Avg / Min by Generation")
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()

def plot_top_groups(name, sorted_arrays, out_png):
    gens, t1, t25, t6_10 = top_group_curves(sorted_arrays, top_k_max=10)
    plt.figure(figsize=(12, 6))
    plt.plot(gens, t1, label="Top 1", linewidth=2.0)
    plt.plot(gens, t25, label="Top 2–5 Avg", linewidth=1.8)
    plt.plot(gens, t6_10, label="Top 6–10 Avg", linewidth=1.8)
    plt.xlabel("Generation"); plt.ylabel("Score")
    plt.title(f"Top-group Curves — {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()

def plot_delta_vs_base(gen, delta_max, delta_avg, label, out_png):
    plt.figure(figsize=(12, 5))
    plt.plot(gen, delta_max, label=f"{label} Max − GraphGA Max", linewidth=1.8)
    plt.plot(gen, delta_avg, label=f"{label} Avg − GraphGA Avg", linewidth=1.8)
    plt.axhline(0.0, color="k", linewidth=1.0, alpha=0.6)
    plt.xlabel("Generation"); plt.ylabel("Score Δ vs GraphGA")
    plt.title(f"Per-generation Difference vs GraphGA — {label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="GraphGA as absolute reference; compare LLM/LLM_alt vs GraphGA")
    ap.add_argument("--graphga", default="data/offspring/amlodipine.csv",
                    help="GraphGA offspring CSV (baseline; requires generation,smiles,score)")
    ap.add_argument("--llm", default="results/DPO_results/ground_truth.csv",
                    help="Method 1 (LLM) CSV")
    ap.add_argument("--llm_alt", default="results/DPO_results/ground_truth_3.csv",
                    help="Method 2 (LLM_alt) CSV; optional")
    ap.add_argument("--outdir", default="analysis_out", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # load
    df_ga = load_scores(args.graphga, "GraphGA")
    if df_ga is None:
        raise SystemExit("GraphGA baseline is required.")
    df_llm = load_scores(args.llm, "LLM")
    df_llm_alt = load_scores(args.llm_alt, "LLM_alt")

    # stats
    stats_map = {}
    sort_map = {}
    for df in [df_ga, df_llm, df_llm_alt]:
        if df is None: 
            continue
        name = df["method"].iloc[0]
        s, sa = per_gen_stats(df)
        stats_map[name] = s
        sort_map[name] = sa

    # merge for aligned comparison (GraphGA is the left table)
    base = stats_map["GraphGA"][["generation", "max", "avg", "min"]].copy()
    base.columns = ["generation", "GraphGA_max", "GraphGA_avg", "GraphGA_min"]
    merged = base.copy()

    for name in ["LLM", "LLM_alt"]:
        if name not in stats_map: 
            continue
        s = stats_map[name][["generation", "max", "avg", "min"]].copy()
        s.columns = ["generation", f"{name}_max", f"{name}_avg", f"{name}_min"]
        merged = pd.merge(merged, s, on="generation", how="outer")

    merged = merged.sort_values("generation").reset_index(drop=True)
    merged.to_csv(os.path.join(args.outdir, "metrics_summary.csv"), index=False)

    # --- deltas vs GraphGA（方法 − GraphGA） ---
    delta_cols = []
    for name in ["LLM", "LLM_alt"]:
        if name not in stats_map: 
            continue
        for stat in ["max", "avg", "min"]:
            mcol = f"{name}_{stat}"
            bcol = f"GraphGA_{stat}"
            dcol = f"delta_{name}_{stat}"
            merged[dcol] = merged[mcol] - merged[bcol]
            delta_cols.append(dcol)

    merged[["generation"] + delta_cols].to_csv(
        os.path.join(args.outdir, "deltas_vs_graphga.csv"), index=False
    )

    # --- winrates vs GraphGA（Max、Avg） ---
    summary_rows = []
    for name in ["LLM", "LLM_alt"]:
        if name not in stats_map: 
            continue
        wr_max, w_m, t_m = winrate_vs_base(merged["GraphGA_max"], merged[f"{name}_max"])
        wr_avg, w_a, t_a = winrate_vs_base(merged["GraphGA_avg"], merged[f"{name}_avg"])

        # 結尾與平均提升（AUC 近似：各代取平均）
        # 注意：對齊 GraphGA 的 generation 範圍來計算
        aligned = merged.dropna(subset=[f"{name}_max", f"{name}_avg", "GraphGA_max", "GraphGA_avg"])
        last_gen = int(aligned["generation"].iloc[-1]) if len(aligned) else np.nan
        last_delta_max = float(aligned[f"{name}_max"].iloc[-1] - aligned["GraphGA_max"].iloc[-1]) if len(aligned) else np.nan
        last_delta_avg = float(aligned[f"{name}_avg"].iloc[-1] - aligned["GraphGA_avg"].iloc[-1]) if len(aligned) else np.nan
        mean_delta_max = float((aligned[f"{name}_max"] - aligned["GraphGA_max"]).mean()) if len(aligned) else np.nan
        mean_delta_avg = float((aligned[f"{name}_avg"] - aligned["GraphGA_avg"]).mean()) if len(aligned) else np.nan

        summary_rows.append({
            "method": name,
            "winrate_max": wr_max, "wins_max": w_m, "total_max": t_m,
            "winrate_avg": wr_avg, "wins_avg": w_a, "total_avg": t_a,
            "last_generation": last_gen,
            "last_delta_max": last_delta_max,
            "last_delta_avg": last_delta_avg,
            "mean_delta_max": mean_delta_max,
            "mean_delta_avg": mean_delta_avg,
        })

        # 繪 delta 圖（Max/Avg）
        if len(aligned):
            plot_delta_vs_base(
                aligned["generation"].to_numpy(),
                (aligned[f"{name}_max"] - aligned["GraphGA_max"]).to_numpy(),
                (aligned[f"{name}_avg"] - aligned["GraphGA_avg"]).to_numpy(),
                label=name,
                out_png=os.path.join(args.outdir, f"delta_max_avg_vs_graphga_{name.lower()}.png"),
            )

    pd.DataFrame(summary_rows).to_csv(os.path.join(args.outdir, "winrate_vs_graphga_summary.csv"), index=False)

    # --- 比較圖（Max/Avg/Min） ---
    plot_comparison(stats_map, os.path.join(args.outdir, "comparison_max_avg_min.png"))

    # --- Top grouping 圖 ---
    plot_top_groups("GraphGA", sort_map["GraphGA"], os.path.join(args.outdir, "top_groups_graphga.png"))
    if "LLM" in sort_map:
        plot_top_groups("LLM", sort_map["LLM"], os.path.join(args.outdir, "top_groups_llm.png"))
    if "LLM_alt" in sort_map:
        plot_top_groups("LLM_alt", sort_map["LLM_alt"], os.path.join(args.outdir, "top_groups_llm_alt.png"))

    print(f"[OK] 完成。輸出位於 {args.outdir}")
    print("  - metrics_summary.csv（各法每代 Max/Avg/Min）")
    print("  - deltas_vs_graphga.csv（方法 − GraphGA 的每代差值）")
    print("  - winrate_vs_graphga_summary.csv（對 GraphGA 的勝率與提升總覽）")
    print("  - comparison_max_avg_min.png / delta_* / top_groups_*.png")

if __name__ == "__main__":
    main()