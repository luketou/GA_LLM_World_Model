import os
from openai import OpenAI

token = "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"
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "",
        },
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ],
    temperature=1,
    top_p=1,
    model=model
)

print(response.choices[0].message.content)

