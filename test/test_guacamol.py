#!/usr/bin/env python3
"""
測試 GuacaMol benchmark 的正確調用方式
"""

from guacamol.standard_benchmarks import similarity

# 測試 celecoxib benchmark
CELECOXIB_SMILES = 'CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F'

try:
    benchmark = similarity(CELECOXIB_SMILES, 'Celecoxib', fp_type='ECFP4', threshold=1.0)
    print(f"Benchmark type: {type(benchmark)}")
    print(f"Benchmark methods: {[attr for attr in dir(benchmark) if not attr.startswith('_')]}")
    
    # 檢查 objective 屬性
    if hasattr(benchmark, 'objective'):
        print(f"Objective type: {type(benchmark.objective)}")
        print(f"Objective methods: {[attr for attr in dir(benchmark.objective) if not attr.startswith('_')]}")
        
        # 測試使用 objective.score 方法
        test_smiles = ["CCO", "CCC", CELECOXIB_SMILES]
        
        if hasattr(benchmark.objective, 'score'):
            print("Testing objective.score method...")
            try:
                scores = []
                for smiles in test_smiles:
                    score = benchmark.objective.score(smiles)
                    scores.append(score)
                    print(f"  {smiles}: {score}")
                print(f"objective.score results: {scores}")
            except Exception as e:
                print(f"objective.score failed: {e}")
                import traceback
                traceback.print_exc()

except Exception as e:
    print(f"Failed to create benchmark: {e}")
    import traceback
    traceback.print_exc()
