#!/usr/bin/env python3
"""
簡單的配置測試腳本
測試 settings.yml 讀取和 API 金鑰設定
"""
import os
import re
import yaml
import pathlib

def test_config_loading():
    print("=== 測試配置讀取 ===")
    
    # 自定義 YAML 載入器
    def env_var_constructor(loader, node):
        value = loader.construct_scalar(node)
        pattern = re.compile(r'\$\{([^}]+)\}')
        match = pattern.search(value)
        if match:
            env_var = match.group(1)
            return os.environ.get(env_var, "")
        return value

    # 註冊自定義構造器
    yaml.SafeLoader.add_constructor('!ENV', env_var_constructor)
    
    # 讀取配置
    config_path = pathlib.Path("config/settings.yml")
    if not config_path.exists():
        print(f"錯誤：找不到配置文件 {config_path}")
        return False
        
    try:
        cfg = yaml.safe_load(config_path.read_text())
        print("✓ 配置文件讀取成功")
        
        # 檢查 LangSmith 配置
        langsmith_config = cfg.get("langsmith", {})
        print(f"LangSmith 啟用: {langsmith_config.get('enabled', False)}")
        print(f"LangSmith 項目: {langsmith_config.get('project', 'N/A')}")
        
        # 檢查 LLM 配置
        llm_config = cfg.get("llm", {})
        print(f"LLM 提供者: {llm_config.get('provider', 'N/A')}")
        print(f"LLM 模型: {llm_config.get('model_name', 'N/A')}")
        api_key = llm_config.get('api_key', '')
        if api_key:
            print(f"LLM API 金鑰: {api_key[:10]}...{api_key[-5:] if len(api_key) > 15 else api_key}")
        else:
            print("LLM API 金鑰: 未設定")
            
        return True
        
    except Exception as e:
        print(f"錯誤：無法讀取配置文件：{e}")
        return False

if __name__ == "__main__":
    success = test_config_loading()
    if success:
        print("\n✓ 配置測試通過")
    else:
        print("\n✗ 配置測試失敗")
