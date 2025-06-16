#!/usr/bin/env python3
"""
檢查配置讀取和工作流程設定
"""
import yaml
import pathlib
import re
import os

# 自定義 YAML 載入器來處理環境變數
def env_var_constructor(loader, node):
    """處理 !ENV 標籤，讀取環境變數"""
    value = loader.construct_scalar(node)
    # 支援 ${VAR_NAME} 格式
    pattern = re.compile(r'\$\{([^}]+)\}')
    match = pattern.search(value)
    if match:
        env_var = match.group(1)
        return os.environ.get(env_var, "")
    return value

# 註冊自定義構造器
yaml.SafeLoader.add_constructor('!ENV', env_var_constructor)

def check_config():
    config_path = pathlib.Path("config/settings.yml")
    if not config_path.exists():
        print("❌ settings.yml 文件不存在")
        return False
    
    try:
        cfg = yaml.safe_load(config_path.read_text())
        print("✅ 成功讀取 settings.yml")
        
        # 檢查工作流程配置
        workflow_config = cfg.get("workflow", {})
        recursion_limit = workflow_config.get("recursion_limit", 50)
        max_iterations = workflow_config.get("max_iterations", 1000)
        
        print(f"📋 工作流程配置:")
        print(f"   遞迴限制: {recursion_limit}")
        print(f"   最大迭代: {max_iterations}")
        
        # 檢查 LangSmith 配置
        langsmith_config = cfg.get("langsmith", {})
        if langsmith_config.get("enabled", False):
            print(f"🔍 LangSmith 追蹤: 已啟用")
            print(f"   專案: {langsmith_config.get('project', 'N/A')}")
            print(f"   API Key: {'已設定' if langsmith_config.get('api_key') else '未設定'}")
        else:
            print(f"🔍 LangSmith 追蹤: 已停用")
        
        # 檢查 LLM 配置
        llm_config = cfg.get("llm", {})
        print(f"🤖 LLM 配置:")
        print(f"   提供者: {llm_config.get('provider', 'N/A')}")
        print(f"   模型: {llm_config.get('model_name', 'N/A')}")
        print(f"   API Key: {'已設定' if llm_config.get('api_key') else '未設定'}")
        
        return True
        
    except Exception as e:
        print(f"❌ 讀取配置文件時出錯: {e}")
        return False

if __name__ == "__main__":
    print("檢查配置設定...")
    check_config()
