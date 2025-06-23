"""
LLM Prompt Templates
系統提示模板和動作提示模板
確保格式化後傳遞給 LLM 的訊息結構正確
"""
from textwrap import dedent
from typing import List, Dict, Any

# TODO: double check generator prompt templates it is trouble 
# 系統提示模板
SYS_TEMPLATE = dedent("""\
You are an expert medicinal chemist with deep knowledge of molecular design and SMILES notation.

Your task is to generate valid SMILES strings for molecules based on a parent molecule and specified chemical modifications.

Key guidelines:
1. Always return chemically valid SMILES strings
2. Apply the requested chemical modifications accurately
3. Maintain drug-likeness properties when possible
4. Ensure the generated molecules are chemically reasonable
5. Return exactly the number of molecules requested

Format your response as a JSON list of SMILES strings:
["SMILES1", "SMILES2", "SMILES3", ...]

Important:
- Only return the JSON list, no explanations or additional text
- Ensure molecular validity and chemical feasibility
- Consider the chemical logic of the transformations
""")

# 動作提示模板
ACTION_TEMPLATE = """Parent molecule: {parent_smiles}

Please generate {num_molecules} new molecules by applying the following chemical modifications:

{actions_description}

Requirements:
- Start with the parent molecule: {parent_smiles}
- Apply the specified modifications to create structurally related molecules
- Ensure all generated SMILES are valid and chemically reasonable
- Return exactly {num_molecules} unique SMILES strings
- Focus on maintaining or improving drug-like properties

Return your response as a JSON list of SMILES strings only, no additional text:
"""


def format_actions_description(actions: List[Dict[str, Any]]) -> str:
    """
    將動作列表格式化為可讀的描述
    """
    descriptions = []
    for i, action in enumerate(actions, 1):
        action_type = action.get("type", "unknown")
        params = action.get("params", {})
        
        if action_type == "add_polar_group":
            group = params.get("group", "unknown")
            position = params.get("position", "")
            desc = f"{i}. Add polar group: {group}"
            if position:
                desc += f" at {position} position"
            descriptions.append(desc)
        elif action_type == "increase_pi_system":
            rings = params.get("rings", 1)
            ring_type = params.get("type", "aromatic ring")
            descriptions.append(f"{i}. Increase π-system by adding {rings} {ring_type}")
        elif action_type == "decrease_molecular_weight":
            if params.get("remove_heavy"):
                atoms = params.get("target_atoms", ["heavy atoms"])
                descriptions.append(f"{i}. Decrease molecular weight by removing {', '.join(atoms)}")
            elif params.get("remove_methyl"):
                descriptions.append(f"{i}. Decrease molecular weight by removing methyl groups")
            else:
                descriptions.append(f"{i}. Decrease molecular weight")
        elif action_type == "swap_heteroatom":
            from_atom = params.get("from", "?")
            to_atom = params.get("to", "?")
            descriptions.append(f"{i}. Swap heteroatom: {from_atom} → {to_atom}")
        elif action_type == "cyclize":
            size = params.get("size", "unknown")
            ring_type = params.get("type", "ring")
            descriptions.append(f"{i}. Cyclize to form {size}-membered {ring_type}")
        elif action_type == "substitute":
            fragment = params.get("fragment", "unknown")
            smiles = params.get("smiles", "")
            desc = f"{i}. Substitute with {fragment} group"
            if smiles:
                desc += f" ({smiles})"
            descriptions.append(desc)
        elif action_type == "add_ring":
            ring_type = params.get("ring_type", "unknown ring")
            smiles = params.get("smiles", "")
            desc = f"{i}. Add {ring_type} ring system"
            if smiles:
                desc += f" ({smiles})"
            descriptions.append(desc)
        else:
            # 通用格式化
            param_str = _format_params(params)
            descriptions.append(f"{i}. Apply {action_type} modification with params: {param_str}")
    
    return "\n".join(descriptions)


def _format_params(params: Dict[str, Any]) -> str:
    """格式化動作參數為可讀字串"""
    if not params:
        return "None"
    
    formatted_parts = []
    for key, value in params.items():
        formatted_parts.append(f"{key}={value}")
    
    return ", ".join(formatted_parts)


def create_llm_messages(parent_smiles: str, actions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    創建發送給 LLM 的完整消息列表
    
    Args:
        parent_smiles: 父分子 SMILES
        actions: 動作列表
        
    Returns:
        LLM 消息列表
    """
    if not actions:
        return [
            {"role": "system", "content": SYS_TEMPLATE},
            {"role": "user", "content": f"Generate a valid SMILES variation of: {parent_smiles}"}
        ]
    
    actions_desc = format_actions_description(actions)
    num_molecules = len(actions)
    
    action_prompt = ACTION_TEMPLATE.format(
        parent_smiles=parent_smiles,
        num_molecules=num_molecules,
        actions_description=actions_desc
    )
    
    return [
        {"role": "system", "content": SYS_TEMPLATE},
        {"role": "user", "content": action_prompt}
    ]


def create_simple_generation_prompt(parent_smiles: str, num_variants: int = 5) -> List[Dict[str, str]]:
    """
    創建簡單的分子生成提示
    
    Args:
        parent_smiles: 父分子 SMILES
        num_variants: 要生成的變體數量
        
    Returns:
        LLM 消息列表
    """
    system_prompt = f"""You are an expert molecular design AI. Generate {num_variants} chemically valid SMILES variations of the given parent molecule.

Requirements:
1. All SMILES must be chemically valid
2. Generate meaningful chemical variations (not random changes)
3. Return EXACTLY a JSON list format: ["SMILES1", "SMILES2", ...]
4. No explanations, only the JSON list"""

    user_prompt = f"""Parent molecule: {parent_smiles}

Generate {num_variants} chemical variations as a JSON list of SMILES strings."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]