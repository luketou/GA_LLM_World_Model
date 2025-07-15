# -*- coding: utf-8 -*-
"""
Action System - 分子修改動作系統
設計理念：
1. 透過 LLM 根據歷史資料選擇要執行的動作
2. 決定在分子的哪個片段或原子上執行動作
3. 先對 SELFIES 每個原子做標籤，再根據標籤執行動作
"""

import json
import random
from typing import List, Dict, Any, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, BRICS
from selfies import encoder, decoder
import logging

logger = logging.getLogger(__name__)

# 支援的原子類型
ATOM_TYPES = ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br']
BOND_TYPES = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]

class ActionSystem:
    """分子修改動作系統"""
    
    def __init__(self, llm_generator=None):
        """
        初始化動作系統
        Args:
            llm_generator: LLM 生成器實例
        """
        self.llm_generator = llm_generator
        self.action_stats = {}  # 記錄動作執行統計
        
        # 定義所有可用的動作
        self.available_actions = {
            'append_atom': self.append_atom,
            'insert_atom': self.insert_atom,
            'delete_atom': self.delete_atom,
            'change_atom': self.change_atom,
            'change_bond_order': self.change_bond_order,
            'delete_cyclic_bond': self.delete_cyclic_bond,
            'add_ring': self.add_ring,
            'crossover': self.crossover,
            'cut': self.cut_molecule
        }
    
    def label_atoms_in_selfies(self, smiles: str) -> Tuple[str, Dict[int, int]]:
        """
        將 SMILES 轉換為 SELFIES 並標記每個原子
        Returns:
            selfies_str: SELFIES 字串
            atom_mapping: 原子索引映射 {selfies_position: rdkit_atom_idx}
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return "", {}
            
            selfies_str = encoder(smiles)
            atom_mapping = {}
            
            # 簡單的映射策略：按順序對應
            for i, atom in enumerate(mol.GetAtoms()):
                atom_mapping[i] = atom.GetIdx()
            
            return selfies_str, atom_mapping
        except Exception as e:
            logger.error(f"Error in label_atoms_in_selfies: {e}")
            return "", {}
    
    def select_action_with_llm(self, 
                              parent_smiles: str, 
                              history_context: Dict = None) -> Tuple[str, Dict]:
        """
        使用 LLM 選擇要執行的動作和參數
        Returns:
            action_name: 要執行的動作名稱
            action_params: 動作參數
        """
        if self.llm_generator is None:
            # 如果沒有 LLM，隨機選擇
            action_name = random.choice(list(self.available_actions.keys()))
            return action_name, self._get_default_params(action_name, parent_smiles)
        
        # 準備 LLM prompt
        prompt = self._create_action_selection_prompt(parent_smiles, history_context)
        
        try:
            response = self.llm_generator.generate_single(prompt)
            action_name, params = self._parse_llm_response(response)
            return action_name, params
        except Exception as e:
            logger.error(f"LLM action selection failed: {e}")
            # Fallback to random selection
            action_name = random.choice(list(self.available_actions.keys()))
            return action_name, self._get_default_params(action_name, parent_smiles)
    
    def execute_action(self, 
                       parent_smiles: str, 
                       action_name: str, 
                       params: Dict = None) -> Optional[str]:
        """
        執行指定的動作
        Returns:
            新的 SMILES 字串，如果失敗則返回 None
        """
        if action_name not in self.available_actions:
            logger.error(f"Unknown action: {action_name}")
            return None
        
        try:
            action_func = self.available_actions[action_name]
            result = action_func(parent_smiles, params or {})
            
            # 更新統計
            self._update_action_stats(action_name, result is not None)
            
            return result
        except Exception as e:
            logger.error(f"Error executing action {action_name}: {e}")
            self._update_action_stats(action_name, False)
            return None
    
    # === 具體的動作實現 ===
    
    def append_atom(self, smiles: str, params: Dict) -> Optional[str]:
        """在分子末端添加原子"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            atom_type = params.get('atom_type', random.choice(ATOM_TYPES))
            bond_type = params.get('bond_type', Chem.BondType.SINGLE)
            position = params.get('position', -1)  # -1 表示末端
            
            # 使用 RDKit 的編輯功能
            em = Chem.EditableMol(mol)
            new_atom_idx = em.AddAtom(Chem.Atom(atom_type))
            
            # 找到要連接的原子
            if position == -1:
                # 找到末端原子（度數為1的原子）
                terminal_atoms = [atom.GetIdx() for atom in mol.GetAtoms() 
                                if atom.GetDegree() == 1]
                if terminal_atoms:
                    target_idx = random.choice(terminal_atoms)
                else:
                    target_idx = 0
            else:
                target_idx = min(position, mol.GetNumAtoms() - 1)
            
            em.AddBond(target_idx, new_atom_idx, bond_type)
            new_mol = em.GetMol()
            
            # 驗證分子
            try:
                Chem.SanitizeMol(new_mol)
                return Chem.MolToSmiles(new_mol)
            except:
                return None
                
        except Exception as e:
            logger.error(f"Error in append_atom: {e}")
            return None
    
    def insert_atom(self, smiles: str, params: Dict) -> Optional[str]:
        """在分子中間插入原子"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            atom_type = params.get('atom_type', random.choice(ATOM_TYPES))
            bond_idx = params.get('bond_idx', None)
            
            # 如果沒有指定鍵，隨機選擇
            if bond_idx is None:
                bonds = list(mol.GetBonds())
                if not bonds:
                    return None
                bond = random.choice(bonds)
                bond_idx = bond.GetIdx()
            
            bond = mol.GetBondWithIdx(bond_idx)
            atom1_idx = bond.GetBeginAtomIdx()
            atom2_idx = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()
            
            # 斷開原有的鍵，插入新原子
            em = Chem.EditableMol(mol)
            em.RemoveBond(atom1_idx, atom2_idx)
            new_atom_idx = em.AddAtom(Chem.Atom(atom_type))
            em.AddBond(atom1_idx, new_atom_idx, bond_type)
            em.AddBond(new_atom_idx, atom2_idx, bond_type)
            
            new_mol = em.GetMol()
            try:
                Chem.SanitizeMol(new_mol)
                return Chem.MolToSmiles(new_mol)
            except:
                return None
                
        except Exception as e:
            logger.error(f"Error in insert_atom: {e}")
            return None
    
    def delete_atom(self, smiles: str, params: Dict) -> Optional[str]:
        """刪除原子"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or mol.GetNumAtoms() <= 1:
                return None
            
            atom_idx = params.get('atom_idx', None)
            degree_filter = params.get('degree', None)  # 根據度數過濾
            
            # 選擇要刪除的原子
            if atom_idx is None:
                candidates = []
                for atom in mol.GetAtoms():
                    if degree_filter is None or atom.GetDegree() == degree_filter:
                        candidates.append(atom.GetIdx())
                
                if not candidates:
                    return None
                atom_idx = random.choice(candidates)
            
            # 刪除原子
            em = Chem.EditableMol(mol)
            
            # 如果刪除的原子度數為2，嘗試連接其鄰居
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetDegree() == 2:
                neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
                if len(neighbors) == 2:
                    # 獲取原有鍵的類型
                    bond1 = mol.GetBondBetweenAtoms(atom_idx, neighbors[0])
                    bond2 = mol.GetBondBetweenAtoms(atom_idx, neighbors[1])
                    # 使用較小的鍵級
                    new_bond_type = min(bond1.GetBondType(), bond2.GetBondType())
                    em.AddBond(neighbors[0], neighbors[1], new_bond_type)
            
            em.RemoveAtom(atom_idx)
            new_mol = em.GetMol()
            
            try:
                Chem.SanitizeMol(new_mol)
                return Chem.MolToSmiles(new_mol)
            except:
                return None
                
        except Exception as e:
            logger.error(f"Error in delete_atom: {e}")
            return None
    
    def change_atom(self, smiles: str, params: Dict) -> Optional[str]:
        """改變原子類型 - 改進版本，考慮價電子和化學合理性"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            atom_idx = params.get('atom_idx', None)
            new_atom_type = params.get('new_atom_type', None)
            
            # 選擇要改變的原子
            if atom_idx is None:
                # 優先選擇不在環中的原子，避免破壞環結構
                non_ring_atoms = [atom.GetIdx() for atom in mol.GetAtoms() 
                                 if not atom.IsInRing()]
                if non_ring_atoms:
                    atom_idx = random.choice(non_ring_atoms)
                else:
                    atom_idx = random.randint(0, mol.GetNumAtoms() - 1)
            
            current_atom = mol.GetAtomWithIdx(atom_idx)
            current_symbol = current_atom.GetSymbol()
            current_degree = current_atom.GetDegree()
            current_formal_charge = current_atom.GetFormalCharge()
            
            # 獲取當前原子的鍵型資訊
            bond_types = []
            for bond in current_atom.GetBonds():
                bond_types.append(bond.GetBondType())
            
            # 選擇新的原子類型 - 考慮化學合理性
            if new_atom_type is None:
                # 根據當前原子的連接情況選擇合適的替換原子
                suitable_atoms = self._get_suitable_replacement_atoms(
                    current_symbol, current_degree, bond_types
                )
                if not suitable_atoms:
                    return None
                new_atom_type = random.choice(suitable_atoms)
            
            # 創建新原子並設置適當的屬性
            new_atom = Chem.Atom(new_atom_type)
            
            # 對於某些原子類型，可能需要調整形式電荷
            if new_atom_type in ['N', 'O'] and current_degree > 2:
                # 可能需要正電荷來保持價電子平衡
                new_atom.SetFormalCharge(1)
            
            # 替換原子
            em = Chem.EditableMol(mol)
            em.ReplaceAtom(atom_idx, new_atom)
            new_mol = em.GetMol()
            
            # 嘗試多種方式來修復分子
            sanitized = False
            
            # 方法1：直接嘗試標準化
            try:
                Chem.SanitizeMol(new_mol)
                sanitized = True
            except:
                pass
            
            # 方法2：嘗試添加/移除氫原子
            if not sanitized:
                try:
                    new_mol = Chem.AddHs(new_mol)
                    Chem.SanitizeMol(new_mol)
                    new_mol = Chem.RemoveHs(new_mol)
                    sanitized = True
                except:
                    pass
            
            # 方法3：嘗試調整鍵級
            if not sanitized and len(bond_types) > 0:
                try:
                    # 如果新原子是 N 或 O，可能需要減少某些鍵的級數
                    if new_atom_type in ['N', 'O', 'F']:
                        em2 = Chem.EditableMol(new_mol)
                        for neighbor in new_mol.GetAtomWithIdx(atom_idx).GetNeighbors():
                            bond = new_mol.GetBondBetweenAtoms(atom_idx, neighbor.GetIdx())
                            if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                                em2.RemoveBond(atom_idx, neighbor.GetIdx())
                                em2.AddBond(atom_idx, neighbor.GetIdx(), Chem.BondType.SINGLE)
                                break
                        new_mol = em2.GetMol()
                        Chem.SanitizeMol(new_mol)
                        sanitized = True
                except:
                    pass
            
            if sanitized:
                return Chem.MolToSmiles(new_mol)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error in change_atom: {e}")
            return None
    
    def _get_suitable_replacement_atoms(self, current_atom: str, degree: int, 
                                      bond_types: List) -> List[str]:
        """根據當前原子的連接情況，返回合適的替換原子類型"""
        
        # 計算總鍵級
        total_bond_order = sum(1 if bt == Chem.BondType.SINGLE else 
                              2 if bt == Chem.BondType.DOUBLE else 
                              3 for bt in bond_types)
        
        suitable_atoms = []
        
        # 根據總鍵級和度數選擇合適的原子
        if total_bond_order <= 4:
            # C 可以形成4個鍵
            if current_atom != 'C':
                suitable_atoms.append('C')
            
            # N 可以形成3個鍵（或4個帶正電荷）
            if total_bond_order <= 3 and current_atom != 'N':
                suitable_atoms.append('N')
            
            # O 可以形成2個鍵（或3個帶正電荷）
            if total_bond_order <= 2 and current_atom != 'O':
                suitable_atoms.append('O')
            
            # S 可以形成2、4或6個鍵
            if total_bond_order <= 6 and current_atom != 'S':
                suitable_atoms.append('S')
            
            # 鹵素通常只形成1個鍵
            if total_bond_order == 1 and degree == 1:
                for halogen in ['F', 'Cl', 'Br']:
                    if current_atom != halogen:
                        suitable_atoms.append(halogen)
        
        return suitable_atoms
    
    def change_bond_order(self, smiles: str, params: Dict) -> Optional[str]:
        """改變鍵級"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            bond_idx = params.get('bond_idx', None)
            new_bond_type = params.get('new_bond_type', None)
            
            # 選擇要改變的鍵
            bonds = list(mol.GetBonds())
            if not bonds:
                return None
                
            if bond_idx is None:
                # 優先選擇非環鍵
                non_ring_bonds = [b for b in bonds if not b.IsInRing()]
                if non_ring_bonds:
                    bond = random.choice(non_ring_bonds)
                else:
                    bond = random.choice(bonds)
            else:
                bond = mol.GetBondWithIdx(bond_idx)
            
            # 選擇新的鍵型
            if new_bond_type is None:
                current_type = bond.GetBondType()
                available_types = [bt for bt in BOND_TYPES if bt != current_type]
                if not available_types:
                    return None
                new_bond_type = random.choice(available_types)
            
            # 修改鍵級
            em = Chem.EditableMol(mol)
            em.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            em.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), new_bond_type)
            new_mol = em.GetMol()
            
            try:
                Chem.SanitizeMol(new_mol)
                return Chem.MolToSmiles(new_mol)
            except:
                return None
                
        except Exception as e:
            logger.error(f"Error in change_bond_order: {e}")
            return None
    
    def delete_cyclic_bond(self, smiles: str, params: Dict) -> Optional[str]:
        """刪除環內鍵"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # 找到所有環內鍵
            ring_info = mol.GetRingInfo()
            ring_bonds = []
            for bond in mol.GetBonds():
                if ring_info.NumBondRings(bond.GetIdx()) > 0:
                    ring_bonds.append(bond.GetIdx())
            
            if not ring_bonds:
                return None
            
            # 選擇要刪除的鍵
            bond_idx = params.get('bond_idx', random.choice(ring_bonds))
            bond = mol.GetBondWithIdx(bond_idx)
            
            # 刪除鍵
            em = Chem.EditableMol(mol)
            em.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            new_mol = em.GetMol()
            
            try:
                Chem.SanitizeMol(new_mol)
                return Chem.MolToSmiles(new_mol)
            except:
                return None
                
        except Exception as e:
            logger.error(f"Error in delete_cyclic_bond: {e}")
            return None
    
    def add_ring(self, smiles: str, params: Dict) -> Optional[str]:
        """添加環結構"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            ring_size = params.get('ring_size', random.choice([5, 6]))
            
            # 簡單的環添加策略：在兩個合適的原子之間添加鍵形成環
            # 找到可以形成環的原子對
            candidates = []
            for i in range(mol.GetNumAtoms()):
                for j in range(i + 2, mol.GetNumAtoms()):  # 至少隔2個原子
                    atom_i = mol.GetAtomWithIdx(i)
                    atom_j = mol.GetAtomWithIdx(j)
                    # 檢查是否已經有鍵
                    if mol.GetBondBetweenAtoms(i, j) is None:
                        # 檢查路徑長度
                        path = Chem.GetShortestPath(mol, i, j)
                        if path and len(path) == ring_size:
                            candidates.append((i, j))
            
            if not candidates:
                return None
            
            # 添加鍵形成環
            atom_i, atom_j = random.choice(candidates)
            em = Chem.EditableMol(mol)
            em.AddBond(atom_i, atom_j, Chem.BondType.SINGLE)
            new_mol = em.GetMol()
            
            try:
                Chem.SanitizeMol(new_mol)
                return Chem.MolToSmiles(new_mol)
            except:
                return None
                
        except Exception as e:
            logger.error(f"Error in add_ring: {e}")
            return None
    
    def crossover(self, smiles: str, params: Dict) -> Optional[str]:
        """分子交叉 - 使用 BRICS 分解"""
        try:
            parent_b_smiles = params.get('parent_b_smiles')
            if not parent_b_smiles:
                # 如果沒有提供第二個分子，使用自身的變體
                parent_b_smiles = smiles
            
            mol_a = Chem.MolFromSmiles(smiles)
            mol_b = Chem.MolFromSmiles(parent_b_smiles)
            
            if mol_a is None or mol_b is None:
                return None
            
            # 使用 BRICS 分解
            try:
                frags_a = list(BRICS.BRICSDecompose(mol_a))
                frags_b = list(BRICS.BRICSDecompose(mol_b))
            except:
                # 如果 BRICS 失敗，嘗試簡單的片段組合
                return self._simple_crossover(mol_a, mol_b)
            
            if not frags_a or not frags_b:
                return self._simple_crossover(mol_a, mol_b)
            
            # 嘗試 BRICS 重組
            try:
                # 使用 BRICS.BRICSBuild 來正確組合片段
                all_frags = [Chem.MolFromSmiles(f) for f in frags_a[:2]] + \
                           [Chem.MolFromSmiles(f) for f in frags_b[:2]]
                
                results = []
                for prod in BRICS.BRICSBuild(all_frags):
                    if prod is not None:
                        try:
                            Chem.SanitizeMol(prod)
                            smiles = Chem.MolToSmiles(prod)
                            if self.mol_ok(smiles):
                                results.append(smiles)
                                if len(results) >= 3:
                                    break
                        except:
                            pass
                
                if results:
                    return random.choice(results)
            except:
                pass
            
            # 如果 BRICS 重組失敗，嘗試簡單組合
            return self._simple_crossover(mol_a, mol_b)
            
        except Exception as e:
            logger.error(f"Error in crossover: {e}")
            return None
    
    def _simple_crossover(self, mol_a: Chem.Mol, mol_b: Chem.Mol) -> Optional[str]:
        """簡單的分子交叉 - 當 BRICS 失敗時的備用方案"""
        try:
            # 簡單地組合兩個分子的部分結構
            if mol_a.GetNumAtoms() > 5 and mol_b.GetNumAtoms() > 5:
                # 各取一半原子
                half_a = mol_a.GetNumAtoms() // 2
                half_b = mol_b.GetNumAtoms() // 2
                
                # 這裡只是返回其中一個分子作為簡單的備用方案
                # 實際應用中可以實現更複雜的片段組合邏輯
                return Chem.MolToSmiles(mol_a) if random.random() > 0.5 else Chem.MolToSmiles(mol_b)
            
            return None
        except:
            return None
    
    def cut_molecule(self, smiles: str, params: Dict) -> Optional[str]:
        """切割分子"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # 找到可切割的鍵（非環單鍵）
            cuttable_bonds = []
            for bond in mol.GetBonds():
                if not bond.IsInRing() and bond.GetBondType() == Chem.BondType.SINGLE:
                    cuttable_bonds.append(bond.GetIdx())
            
            if not cuttable_bonds:
                return None
            
            # 選擇切割位置
            bond_idx = params.get('bond_idx', random.choice(cuttable_bonds))
            
            # 切割分子
            frags = Chem.FragmentOnBonds(mol, [bond_idx], addDummies=True)
            frag_mols = Chem.GetMolFrags(frags, asMols=True)
            
            if frag_mols:
                # 返回最大的片段
                largest_frag = max(frag_mols, key=lambda m: m.GetNumAtoms())
                # 移除虛擬原子
                largest_frag = Chem.DeleteSubstructs(largest_frag, Chem.MolFromSmarts('[#0]'))
                return Chem.MolToSmiles(largest_frag)
            
            return None
            
        except Exception as e:
            logger.error(f"Error in cut_molecule: {e}")
            return None
    
    # === 輔助方法 ===
    
    def mol_ok(self, smiles: str) -> bool:
        """檢查分子是否合法"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            # 基本檢查
            if mol.GetNumAtoms() < 5 or mol.GetNumAtoms() > 50:
                return False
            
            # 檢查分子是否可以標準化
            try:
                Chem.SanitizeMol(mol)
                return True
            except:
                return False
                
        except:
            return False
    
    def ring_ok(self, smiles: str) -> bool:
        """檢查環結構是否合理"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            ring_info = mol.GetRingInfo()
            
            # 檢查環大小
            for ring in ring_info.AtomRings():
                if len(ring) < 3 or len(ring) > 8:
                    return False
            
            return True
            
        except:
            return False
    
    def _create_action_selection_prompt(self, smiles: str, history: Dict = None) -> str:
        """創建 LLM 動作選擇提示"""
        selfies_str, atom_mapping = self.label_atoms_in_selfies(smiles)
        
        prompt = f"""Given the molecule:
SMILES: {smiles}
SELFIES: {selfies_str}
Atom labels: {atom_mapping}

Available actions:
{json.dumps(list(self.available_actions.keys()), indent=2)}

Select the best action and parameters.

Response format:
{{
    "action": "action_name",
    "params": {{}},
    "reasoning": "brief explanation"
}}
"""
        
        if history:
            prompt += f"\nHistory context: {json.dumps(history, indent=2)}"
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> Tuple[str, Dict]:
        """解析 LLM 回應"""
        try:
            # 嘗試提取 JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                action = data.get('action', 'change_atom')
                params = data.get('params', {})
                return action, params
        except:
            pass
        
        # Fallback
        return 'change_atom', {}
    
    def _get_default_params(self, action_name: str, smiles: str) -> Dict:
        """獲取動作的預設參數"""
        default_params = {
            'append_atom': {'atom_type': 'C', 'bond_type': Chem.BondType.SINGLE},
            'insert_atom': {'atom_type': 'C'},
            'delete_atom': {'degree': 1},
            'change_atom': {},
            'change_bond_order': {},
            'delete_cyclic_bond': {},
            'add_ring': {'ring_size': 6},
            'crossover': {'parent_b_smiles': smiles},  # 需要第二個分子
            'cut': {}
        }
        return default_params.get(action_name, {})
    
    def _update_action_stats(self, action_name: str, success: bool):
        """更新動作統計"""
        if action_name not in self.action_stats:
            self.action_stats[action_name] = {
                'attempts': 0,
                'successes': 0,
                'success_rate': 0.0
            }
        
        stats = self.action_stats[action_name]
        stats['attempts'] += 1
        if success:
            stats['successes'] += 1
        stats['success_rate'] = stats['successes'] / stats['attempts'] if stats['attempts'] > 0 else 0
    
    def get_action_stats(self) -> Dict:
        """獲取動作統計資訊"""
        return self.action_stats


# 使用範例
if __name__ == "__main__":
    # 初始化
    action_system = ActionSystem()
    
    # 測試分子
    test_smiles = "CC(C)C1=CC=CC=C1"
    
    # 執行動作
    print(f"Original: {test_smiles}")
    print("=" * 50)
    
    # 測試各種動作
    test_actions = ['append_atom', 'change_atom', 'add_ring', 'insert_atom', 
                   'delete_atom', 'change_bond_order', 'crossover']
    
    for action in test_actions:
        print(f"\nTesting {action}:")
        result = action_system.execute_action(test_smiles, action)
        if result:
            print(f"  Result: {result}")
            print(f"  Valid: {'✓' if action_system.mol_ok(result) else '✗'}")
            if action_system.ring_ok(result):
                print(f"  Ring structure: OK")
        else:
            print(f"  Failed to generate valid molecule")

    # 顯示統計
    print("\n" + "=" * 50)
    print("Action Statistics:")
    stats = action_system.get_action_stats()
    for action, stat in stats.items():
        print(f"  {action}: {stat['success_rate']:.1%} success ({stat['successes']}/{stat['attempts']})")
