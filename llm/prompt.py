"""
LLM Prompt Templates - Enhanced for Reliable SMILES Generation with Token Markers
強化版提示模板，使用特定token標記解決 LLM 生成失敗問題
"""
from textwrap import dedent
from typing import List, Dict, Any

# 強化系統提示 - 使用特定token標記
ENHANCED_SYSTEM_TEMPLATE = dedent("""\
You are an expert medicinal chemist specializing in SMILES notation and molecular design.
Your task is to generate valid SMILES strings based on chemical modifications.

CRITICAL REQUIREMENTS:
1. ALWAYS wrap each SMILES string with <SMILES> and </SMILES> tokens
2. Generate one SMILES per line
3. NO explanations, NO thinking process, NO additional text
4. Every SMILES must be chemically valid and unique
5. Apply the requested modifications systematically

FORMAT EXAMPLE:
<SMILES>c1ccc(Cl)cc1</SMILES>
<SMILES>c1ccc(Br)cc1</SMILES>
<SMILES>c1ccc(F)cc1</SMILES>

FORBIDDEN:
- Do NOT start with <think> or any thinking tags
- Do NOT use words like "Okay," "First," "Now," etc.
- Do NOT provide explanations or descriptions
- Do NOT use "Modification" or similar words
- Do NOT forget the <SMILES></SMILES> wrapper tokens
""")

# Few-shot 範例提示模板 - 使用token標記
FEW_SHOT_TEMPLATE = dedent("""\
Parent: c1ccccc1
Modifications: Add halogen substituents
Response:
<SMILES>c1ccc(Cl)cc1</SMILES>
<SMILES>c1ccc(Br)cc1</SMILES>
<SMILES>c1ccc(F)cc1</SMILES>

Parent: CCO
Modifications: Add functional groups
Response:
<SMILES>CCN</SMILES>
<SMILES>CCC</SMILES>
<SMILES>CC(C)O</SMILES>

Parent: {parent_smiles}
Modifications: {modifications}
Response:""")

def format_pubchem_info(pubchem_data: dict) -> str:
    """
    將 PubChem 查詢結果格式化為 LLM prompt 可讀字串
    """
    if not pubchem_data or 'error' in pubchem_data:
        return "(No PubChem info found)"
    lines = [
        f"PubChem CID: {pubchem_data.get('cid', '-')}",
        f"IUPAC Name: {pubchem_data.get('iupac_name', '-')}\nMolecular Formula: {pubchem_data.get('molecular_formula', '-')}\nMolecular Weight: {pubchem_data.get('molecular_weight', '-')}\nCanonical SMILES: {pubchem_data.get('canonical_smiles', '-')}\nXLogP: {pubchem_data.get('xlogp', '-')}\nH-bond Donor Count: {pubchem_data.get('h_bond_donor_count', '-')}\nH-bond Acceptor Count: {pubchem_data.get('h_bond_acceptor_count', '-')}\nPubChem URL: {pubchem_data.get('pubchem_url', '-')}"
    ]
    return '\n'.join(lines)

def create_enhanced_llm_messages(parent_smiles: str, actions: List[Dict[str, Any]], pubchem_data: dict = None) -> List[Dict[str, str]]:
    """
    創建強化版 LLM 消息，使用 <SMILES></SMILES> token 標記，並可選擇加入 PubChem 資訊
    """
    if not actions:
        modifications = "Generate chemical variations"
    else:
        modifications = _create_clear_modifications_text(actions)
    
    pubchem_info_block = f"\n\n[PubChem Information for Parent SMILES]\n{format_pubchem_info(pubchem_data)}" if pubchem_data else ""
    # 使用 Few-shot 範例
    user_prompt = FEW_SHOT_TEMPLATE.format(
        parent_smiles=parent_smiles,
        modifications=modifications
    ) + pubchem_info_block
    
    return [
        {"role": "system", "content": ENHANCED_SYSTEM_TEMPLATE},
        {"role": "user", "content": user_prompt}
    ]

def _create_clear_modifications_text(actions: List[Dict[str, Any]]) -> str:
    """創建清晰的修改描述，避免複雜語言"""
    modifications = []
    
    for action in actions:
        action_type = action.get("type", "unknown")
        action_name = action.get("name", "unnamed")
        
        # 簡化動作描述
        if action_type == "substitute":
            modifications.append(f"Add {action_name.replace('add_', '')}")
        elif action_type == "scaffold_swap":
            modifications.append(f"Replace with {action_name.replace('swap_to_', '')}")
        elif action_type == "cyclization":
            modifications.append("Form ring")
        elif action_type == "ring_opening":
            modifications.append("Open ring")
        else:
            modifications.append(action_name.replace('_', ' '))
    
    return "; ".join(modifications[:10])  # 限制長度避免過度複雜

def create_simple_generation_prompt(parent_smiles: str, num_variants: int = 5, pubchem_data: dict = None) -> List[Dict[str, str]]:
    """
    創建簡單可靠的分子生成提示 - 使用token標記，並可選擇加入 PubChem 資訊
    """
    system_prompt = f"""You are a SMILES generator. Generate {num_variants} valid SMILES variations.\nWRAP each SMILES with <SMILES></SMILES> tokens.\nOne SMILES per line. NO other text allowed.\n\nEXAMPLE:\n<SMILES>CCO</SMILES>\n<SMILES>CCN</SMILES>"""
    pubchem_info_block = f"\n\n[PubChem Information for Parent SMILES]\n{format_pubchem_info(pubchem_data)}" if pubchem_data else ""
    user_prompt = f'Parent: {parent_smiles}\nGenerate {num_variants} variations:' + pubchem_info_block
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

def create_fallback_prompt(parent_smiles: str, num_variants: int = 5, pubchem_data: dict = None) -> List[Dict[str, str]]:
    """創建最簡化的後備提示 - 使用token標記，並可選擇加入 PubChem 資訊"""
    pubchem_info_block = f"\n\n[PubChem Information for Parent SMILES]\n{format_pubchem_info(pubchem_data)}" if pubchem_data else ""
    return [
        {
            "role": "system", 
            "content": f"Generate {num_variants} valid SMILES. Use format: <SMILES>SMILES_STRING</SMILES>"
        },
        {
            "role": "user", 
            "content": f"{parent_smiles} -> variations" + pubchem_info_block
        }
    ]