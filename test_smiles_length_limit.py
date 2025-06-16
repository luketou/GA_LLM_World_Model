#!/usr/bin/env python3
"""
測試 SMILES 長度限制功能
"""

import sys
import pathlib
import yaml

# 添加專案根目錄到 Python 路徑
project_root = pathlib.Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_config_loading():
    """測試配置檔案是否正確載入長度限制"""
    cfg = yaml.safe_load(pathlib.Path("config/settings.yml").read_text())
    max_length = cfg.get("workflow", {}).get("max_smiles_length", 100)
    print(f"配置的最大 SMILES 長度: {max_length}")
    assert max_length == 100, f"期望長度為 100，實際為 {max_length}"
    print("✓ 配置載入測試通過")

def test_llm_generator_initialization():
    """測試 LLM 生成器是否正確初始化長度限制"""
    from llm.generator import LLMGenerator
    
    # 測試默認值
    generator = LLMGenerator()
    assert hasattr(generator, 'max_smiles_length'), "LLMGenerator 應該有 max_smiles_length 屬性"
    assert generator.max_smiles_length == 100, f"期望默認長度為 100，實際為 {generator.max_smiles_length}"
    print("✓ LLM 生成器默認初始化測試通過")
    
    # 測試自定義值
    custom_generator = LLMGenerator(max_smiles_length=80)
    assert custom_generator.max_smiles_length == 80, f"期望自定義長度為 80，實際為 {custom_generator.max_smiles_length}"
    print("✓ LLM 生成器自定義初始化測試通過")

def test_smiles_filtering():
    """測試 SMILES 過濾功能"""
    from llm.generator import LLMGenerator
    
    generator = LLMGenerator(max_smiles_length=50)
    
    # 模擬 SMILES 列表，包含短的和長的
    test_smiles = [
        "CCO",  # 短 (3字符)
        "C1=CC=CC=C1",  # 中等 (10字符)
        "C" * 60,  # 長 (60字符，超過限制50)
        "CC(C)C(=O)N1CCC2(CC1)CN(C(=O)C3=CC=CC=C3)C2",  # 中等長度 (44字符)
    ]
    
    # 測試基本檢查功能
    filtered = generator._basic_smiles_check(test_smiles)
    
    print(f"原始 SMILES 數量: {len(test_smiles)}")
    print(f"過濾後數量: {len(filtered)}")
    
    # 檢查是否過濾掉了過長的 SMILES
    for smiles in filtered:
        assert len(smiles) <= 50, f"過濾後仍有過長的 SMILES: {smiles} (長度: {len(smiles)})"
    
    print("✓ SMILES 過濾測試通過")

def test_example_long_smiles():
    """測試處理使用者提到的長 SMILES 例子"""
    long_smiles = "C1CC2C3CCC4CC(OC5OC(CO)C(O)C(O)C5OC5OC(CO)C(O)C(O)C5O)CCC4(C)C3CCC2(C)C1OCC1=CC=CC=C1C1=CC=CC=C1C1=CN=CC=C1C1=CC=CC=C1S(=O)(=O)C(=O)O"
    
    print(f"測試的長 SMILES 長度: {len(long_smiles)}")
    print(f"SMILES: {long_smiles}")
    
    # 測試是否會被過濾掉（默認限制是100）
    from llm.generator import LLMGenerator
    generator = LLMGenerator(max_smiles_length=100)
    
    filtered = generator._basic_smiles_check([long_smiles])
    
    if len(long_smiles) > 100:
        assert len(filtered) == 0, "過長的 SMILES 應該被過濾掉"
        print("✓ 長 SMILES 被正確過濾")
    else:
        assert len(filtered) == 1, "符合長度限制的 SMILES 不應該被過濾"
        print("✓ 符合長度的 SMILES 未被過濾")

if __name__ == "__main__":
    print("=== 測試 SMILES 長度限制功能 ===\n")
    
    try:
        test_config_loading()
        test_llm_generator_initialization()
        test_smiles_filtering()
        test_example_long_smiles()
        
        print("\n🎉 所有測試通過！")
        print("\n配置說明：")
        print("- 在 config/settings.yml 中設定 workflow.max_smiles_length 來控制最大長度")
        print("- 建議值：50-150，根據你的需求調整")
        print("- 100 是一個適中的值，可以防止過度複雜的分子")
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
