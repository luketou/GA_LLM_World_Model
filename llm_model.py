from vllm import LLM, SamplingParams

# 指定 HuggingFace 模型名稱
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

# 初始化 LLM
llm = LLM(model=model_name)

# 設定推理參數
sampling_params = SamplingParams(temperature=0.7, max_tokens=128)

# 測試 prompt
prompt = "請簡單介紹一下深度學習。"

# 執行推理
outputs = llm.generate([prompt], sampling_params)

# 顯示結果
for output in outputs:
    print(output.outputs[0].text)