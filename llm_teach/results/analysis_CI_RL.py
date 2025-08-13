import numpy as np
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, average_precision_score
from sklearn.utils import resample
import pandas as pd

# y: oracle scores, c_old/c_new: critic scores (同一批樣本，成對比較)
def metrics(y, c, topk=10, top_pct=0.05, n_boot=1000, rng=0):
    y = np.asarray(y).ravel()
    c = np.asarray(c).ravel()
    assert y.shape == c.shape

    # 排序一致性
    spearman = stats.spearmanr(c, y).correlation
    kendall = stats.kendalltau(c, y).correlation

    # 校準
    mae = mean_absolute_error(y, c)
    rmse = np.sqrt(mean_squared_error(y, c))
    r2 = r2_score(y, c)

    # Calibration slope/intercept
    slope, intercept, _, _, _ = stats.linregress(c, y)

    # ECE（等頻分箱）
    q = np.quantile(c, np.linspace(0,1,11))
    ece = 0.0
    for i in range(10):
        mask = (c >= q[i]) & (c <= q[i+1]) if i<9 else (c > q[i]) & (c <= q[i+1])
        if mask.sum() == 0: 
            continue
        bin_pred = c[mask].mean()
        bin_true = y[mask].mean()
        ece += (mask.mean()) * abs(bin_pred - bin_true)

    # Isotonic regression（可校準性上限）
    ir = IsotonicRegression(out_of_bounds='clip').fit(c, y)
    y_iso = ir.predict(c)
    r2_iso = r2_score(y, y_iso)

    # 選擇能力
    k = min(topk, len(y))
    idx_pred = np.argsort(-c)[:k]
    idx_true = np.argsort(-y)[:k]
    precision_at_k = np.intersect1d(idx_pred, idx_true).size / k

    # NDCG@k
    def dcg(vals): 
        return np.sum((2**vals - 1) / np.log2(np.arange(2, len(vals)+2)))
    rel = y[np.argsort(-c)[:k]]
    ideal = np.sort(y)[::-1][:k]
    ndcg = dcg(rel) / (dcg(ideal) + 1e-12)

    # EF@p%
    p = max(1, int(np.ceil(len(y) * top_pct)))
    top_true = set(np.argsort(-y)[:p])
    got = len(top_true.intersection(set(np.argsort(-c)[:p])))
    ef = got / (p * top_pct)  # 富集倍數

    # AUC-PR（Top-quantile 當正類）
    y_bin = np.zeros_like(y)
    y_bin[np.argsort(-y)[:p]] = 1
    auc_pr = average_precision_score(y_bin, c)

    # bootstrap CI for Spearman/Kendall
    rng = np.random.default_rng(rng)
    s_list, k_list = [], []
    for _ in range(n_boot):
        idx = resample(np.arange(len(y)), replace=True, random_state=int(rng.integers(1e9)))
        s_list.append(stats.spearmanr(c[idx], y[idx]).correlation)
        k_list.append(stats.kendalltau(c[idx], y[idx]).correlation)
    s_ci = np.percentile(s_list, [2.5, 97.5])
    k_ci = np.percentile(k_list, [2.5, 97.5])

    return dict(
        spearman=spearman, spearman_ci=tuple(s_ci),
        kendall=kendall, kendall_ci=tuple(k_ci),
        mae=mae, rmse=rmse, r2=r2, cal_slope=slope, cal_intercept=intercept, ece=ece, r2_isotonic=r2_iso,
        precision_at_k=precision_at_k, ndcg=ndcg, ef_at_pct=ef, auc_pr=auc_pr
    )

# 檢定：前後差異（以 Spearman 為例，用 bootstrap 差值 CI）
def compare(y, cold, cnew):
    m_old = metrics(y, cold)
    m_new = metrics(y, cnew)
    # 差值 CI（Spearman）
    diffs = []
    for _ in range(1000):
        idx = resample(np.arange(len(y)), replace=True)
        d = stats.spearmanr(cnew[idx], y[idx]).correlation - stats.spearmanr(cold[idx], y[idx]).correlation
        diffs.append(d)
    ci = np.percentile(diffs, [2.5, 97.5])
    return m_old, m_new, {"delta_spearman": np.mean(diffs), "delta_ci": tuple(ci)}

if __name__ == "__main__":
    old_csv_path = "/home/luketou/GA_LLM_World_Model/llm_teach/results/muvera_rl_critic_results/amlodipine_old.csv"
    new_csv_path = "/home/luketou/GA_LLM_World_Model/llm_teach/results/muvera_rl_critic_results/amlodipine.csv"

    df_old = pd.read_csv(old_csv_path)
    df_new = pd.read_csv(new_csv_path)

    min_len = min(len(df_old), len(df_new))
    y = df_old["score"].iloc[:min_len]
    cold = df_old["critic_score"].iloc[:min_len]
    cnew = df_new["critic_score"].iloc[:min_len]

    m_old, m_new, delta = compare(y, cold, cnew)
    print("Old metrics:", m_old)
    print("New metrics:", m_new)
    print("Delta CI:", delta)