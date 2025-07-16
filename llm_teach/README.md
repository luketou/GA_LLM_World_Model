# 使用 LLM 協助分子篩選能力測試

## 核心理念
本專案旨在測試大型語言模型（LLM）是否具備辨識分子在特定任務下潛力高低的能力。

## 工作流程
1. 利用 Guacamol 基準的 Graph GA，每一代生成 100 個分子，並以 Guacamol 進行評分。
2. 將每代 100 個分子的 SMILES、任務目標及描述傳遞給 LLM（注意：**不可洩漏 Oracle 評分成績**），由 LLM 根據資訊篩選出 10 個其認為最具潛力、分數可能最高的分子。
3. 將 Graph GA 演化過程的 max/avg 分數繪製成折線圖，並在同一圖上標示每代由 LLM 選出的 10 個分子的 max/avg Oracle 分數。

## 詳細說明
1. 使用 vllm 推理 [deepseek-ai/DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B)。
2. Graph GA 每代結果儲存於 `results/graphga.csv`，格式：`generation,smiles,score`。
3. LLM 每代篩選的 10 個分子儲存於 `results/llm.csv`，格式：`generation,smiles,oracle`。
4. Guacamol 基準範例程式碼如下：
    ```python
    from guacamol.standard_benchmarks import (
        hard_osimertinib, hard_fexofenadine, ranolazine_mpo,
        amlodipine_rings, sitagliptin_replacement, zaleplon_with_other_formula,
        median_camphor_menthol, median_tadalafil_sildenafil, similarity,
        perindopril_rings, hard_cobimetinib, qed_benchmark, logP_benchmark,
        tpsa_benchmark, cns_mpo, scaffold_hop, decoration_hop, weird_physchem,
        isomers_c11h24, isomers_c9h10n2o2pf2cl, valsartan_smarts
    )
    ```

## 任務與基準對應
可用的任務名稱與 Guacamol 基準函式對應如下：

```python
TASK_BENCHMARK_MAPPING = {
    # MPO 任務
    'osimertinib': hard_osimertinib,
    'fexofenadine': hard_fexofenadine,
    'ranolazine': ranolazine_mpo,
    'amlodipine': amlodipine_rings,
    'perindopril': perindopril_rings,
    'sitagliptin': sitagliptin_replacement,
    'zaleplon': zaleplon_with_other_formula,
    'cobimetinib': hard_cobimetinib,
    'qed': qed_benchmark,
    'cns_mpo': cns_mpo,
    'scaffold_hop': scaffold_hop,
    'decoration_hop': decoration_hop,
    'weird_physchem': weird_physchem,
    'valsartan_smarts': valsartan_smarts,

    # 中位數任務
    'median1': median_camphor_menthol,
    'median2': median_tadalafil_sildenafil,

    # 同分異構物任務
    'isomer_c11h24': isomers_c11h24,
    'isomer_c9h10n2o2pf2cl': isomers_c9h10n2o2pf2cl,

    # 性質目標
    'logp_2.5': lambda: logP_benchmark(target_value=2.5),
    'tpsa_100': lambda: tpsa_benchmark(target_value=100),

    # 重新發現任務
    'celecoxib': lambda: similarity(
        CELECOXIB_SMILES, 'Celecoxib', fp_type='ECFP4', threshold=1.0),
    'troglitazone': lambda: similarity(
        TROGLITAZONE_SMILES, 'Troglitazone', fp_type='ECFP4', threshold=1.0),
    'thiothixene': lambda: similarity(
        THIOTHIZENE_SMILES, 'Thiothixene', fp_type='ECFP4', threshold=1.0)
}
```

## 備註
- Graph GA 會繪製 max/avg 曲線，並標示每代 LLM 選出的前十名分子。
