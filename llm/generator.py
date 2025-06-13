from typing import List, Dict
from langchain_openai import ChatOpenAI
from langsmith import traceable
from .prompt_templates import SYS_TEMPLATE, ACTION_TEMPLATE
from rdkit import Chem  # 驗證

class LLMGenerator:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.7)

    @traceable(name="LLM::generate_batch")
    def generate_batch(self,
                       parent_smiles: str,
                       actions: List[Dict]) -> List[str]:
        messages = [{"role": "system",
                     "content": SYS_TEMPLATE.format(smiles=parent_smiles)}]
        for a in actions:
            messages.append({"role": "user",
                             "content": ACTION_TEMPLATE.format(**a)})
        reply = self.llm(messages)
        import json; smis = json.loads(reply.content)
        # rudimentary validation
        return [s for s in smis if Chem.MolFromSmiles(s)]