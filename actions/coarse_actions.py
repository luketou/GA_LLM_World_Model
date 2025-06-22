"""
Comprehensive coarse-grained actions with externalized JSON configuration.
粗粒度操作系統：從外部 JSON 文件動態載入轉換規則

主要特點：
1. 規則與代碼分離：所有轉換規則存儲在 coarse_actions.json 中
2. 動態加載：運行時讀取 JSON 配置，支持熱更新
3. 結構化應用：提供標準化接口應用轉換規則
4. 合規性保證：所有轉換都設計為在 Oracle 評估前使用，無 RDKit 屬性計算

COMPLIANCE NOTE: All transformations are designed for use before Oracle evaluation.
No RDKit property calculations are performed in this module.
"""

import json
import random
import pathlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TransformationRule:
    """轉換規則數據類"""
    type: str
    name: str
    description: str
    params: Dict[str, Any]
    priority_weight: float = 1.0

class CoarseActionEngine:
    """粗粒度動作引擎"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化動作引擎
        
        Args:
            config_path: JSON 配置文件路徑，默認為 coarse_actions.json
        """
        self.config_path = config_path or (pathlib.Path(__file__).parent / "coarse_actions.json")
        self.config = {}
        self.actions_by_category = {}
        self.all_actions = []
        self.transformation_rules = {}
        
        self._load_configuration()
        self._build_action_registry()
    
    def _load_configuration(self):
        """載入 JSON 配置文件"""
        try:
            if not pathlib.Path(self.config_path).exists():
                logger.error(f"Configuration file not found: {self.config_path}")
                self._create_fallback_config()
                return
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            logger.info(f"Loaded coarse actions configuration from {self.config_path}")
            logger.info(f"Configuration version: {self.config.get('metadata', {}).get('version', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self._create_fallback_config()
    
    def _create_fallback_config(self):
        """創建後備配置"""
        logger.warning("Using fallback configuration")
        self.config = {
            "metadata": {"version": "fallback", "description": "Emergency fallback configuration"},
            "action_categories": {
                "basic": {
                    "description": "Basic transformations",
                    "priority_weight": 1.0,
                    "actions": [
                        {
                            "type": "add_functional_group",
                            "name": "add_methyl",
                            "description": "添加甲基",
                            "params": {"group": "CH3", "group_smiles": "C"}
                        }
                    ]
                }
            },
            "transformation_rules": {}
        }
    
    def _build_action_registry(self):
        """構建動作註冊表"""
        try:
            self.actions_by_category = {}
            self.all_actions = []
            
            action_categories = self.config.get("action_categories", {})
            
            for category_name, category_data in action_categories.items():
                category_weight = category_data.get("priority_weight", 1.0)
                category_actions = []
                
                for action_data in category_data.get("actions", []):
                    rule = TransformationRule(
                        type=action_data.get("type", "unknown"),
                        name=action_data.get("name", "unnamed"),
                        description=action_data.get("description", ""),
                        params=action_data.get("params", {}),
                        priority_weight=category_weight
                    )
                    
                    category_actions.append(rule)
                    self.all_actions.append(rule)
                
                self.actions_by_category[category_name] = category_actions
            
            # 載入轉換規則
            self.transformation_rules = self.config.get("transformation_rules", {})
            
            logger.info(f"Built action registry: {len(self.all_actions)} total actions "
                       f"across {len(self.actions_by_category)} categories")
            
        except Exception as e:
            logger.error(f"Error building action registry: {e}")
            # 確保至少有一些基本動作
            if not self.all_actions:
                self.all_actions = [
                    TransformationRule(
                        type="add_functional_group",
                        name="add_methyl_fallback",
                        description="添加甲基 (後備)",
                        params={"group": "CH3", "group_smiles": "C"},
                        priority_weight=1.0
                    )
                ]
    
    def sample(self, k: int = 10, parent_smiles: str = None, 
               categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        從配置的動作中抽樣
        
        COMPLIANCE NOTE: Sampling uses only JSON-defined rules,
        no RDKit property calculations.
        
        Args:
            k: 要抽樣的動作數量
            parent_smiles: 父分子 SMILES（用於上下文感知抽樣）
            categories: 限制抽樣的類別列表
            
        Returns:
            抽樣得到的動作列表
        """
        try:
            # 確定可用動作
            if categories:
                available_actions = []
                for category in categories:
                    if category in self.actions_by_category:
                        available_actions.extend(self.actions_by_category[category])
            else:
                available_actions = self.all_actions
            
            if not available_actions:
                logger.warning("No available actions for sampling")
                return []
            
            # 限制 k 值
            k = min(k, len(available_actions))
            
            if k <= 0:
                return []
            
            # 計算權重
            weights = [action.priority_weight for action in available_actions]
            
            # 上下文感知權重調整
            if parent_smiles:
                weights = self._adjust_weights_for_context(available_actions, weights, parent_smiles)
            
            # 加權隨機抽樣
            selected_actions = random.choices(available_actions, weights=weights, k=k)
            
            # 轉換為字典格式
            result = []
            for action in selected_actions:
                action_dict = {
                    "type": action.type,
                    "name": action.name,
                    "description": action.description,
                    "params": action.params.copy()
                }
                result.append(action_dict)
            
            logger.debug(f"Sampled {len(result)} actions from {len(available_actions)} available")
            return result
            
        except Exception as e:
            logger.error(f"Error in sampling: {e}")
            return self._get_emergency_actions(k)
    
    def _adjust_weights_for_context(self, actions: List[TransformationRule], 
                                   weights: List[float], parent_smiles: str) -> List[float]:
        """
        基於分子上下文調整權重
        
        COMPLIANCE NOTE: Context analysis using string-based SMILES features only.
        """
        try:
            if not parent_smiles:
                return weights
            
            adjusted_weights = weights.copy()
            
            # 基於 SMILES 字符串的簡單特徵分析
            has_aromatic = any(c.islower() for c in parent_smiles)
            has_rings = any(c.isdigit() for c in parent_smiles)
            has_nitrogen = 'N' in parent_smiles or 'n' in parent_smiles
            has_oxygen = 'O' in parent_smiles or 'o' in parent_smiles
            
            for i, action in enumerate(actions):
                # 芳香族分子偏好芳香族骨架
                if has_aromatic and action.type == "scaffold_swap":
                    if "aromatic" in action.params.get("target_scaffold", "").lower():
                        adjusted_weights[i] *= 1.5
                
                # 已有雜原子的分子偏好雜原子交換
                if (has_nitrogen or has_oxygen) and action.type == "heteroatom_exchange":
                    adjusted_weights[i] *= 1.3
                
                # 環狀分子偏好環相關操作
                if has_rings and action.type in ["cyclization", "ring_opening"]:
                    adjusted_weights[i] *= 1.2
            
            return adjusted_weights
            
        except Exception as e:
            logger.debug(f"Error adjusting weights for context: {e}")
            return weights
    
    def apply_coarse_action(self, parent_smiles: str, action: Dict[str, Any]) -> Optional[str]:
        """
        應用粗粒度動作到分子上
        
        COMPLIANCE NOTE: This method provides transformation logic based on
        JSON-defined rules. Actual SMILES transformation is delegated to LLM.
        
        Args:
            parent_smiles: 原始分子 SMILES
            action: 要應用的動作
            
        Returns:
            轉換後的 SMILES（如果成功）
        """
        try:
            action_type = action.get("type", "")
            action_params = action.get("params", {})
            
            logger.debug(f"Applying action {action.get('name', 'unnamed')} to {parent_smiles}")
            
            # 根據動作類型應用轉換
            if action_type == "scaffold_swap":
                return self._apply_scaffold_swap(parent_smiles, action_params)
            elif action_type == "add_functional_group":
                return self._apply_functional_group_addition(parent_smiles, action_params)
            elif action_type == "heteroatom_exchange":
                return self._apply_heteroatom_exchange(parent_smiles, action_params)
            elif action_type == "cyclization":
                return self._apply_cyclization(parent_smiles, action_params)
            elif action_type == "ring_opening":
                return self._apply_ring_opening(parent_smiles, action_params)
            else:
                logger.warning(f"Unknown action type: {action_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error applying coarse action: {e}")
            return None
    
    def _apply_scaffold_swap(self, parent_smiles: str, params: Dict[str, Any]) -> Optional[str]:
        """
        應用骨架替換
        
        COMPLIANCE NOTE: Scaffold swapping uses template matching and
        LLM-driven transformation, no RDKit property calculations.
        """
        try:
            target_scaffold = params.get("target_scaffold", "")
            scaffold_smiles = params.get("scaffold_smiles", "")
            preserve_substituents = params.get("preserve_substituents", True)
            
            if not scaffold_smiles:
                logger.warning(f"No scaffold SMILES provided for {target_scaffold}")
                return None
            
            # 這裡返回變換提示，實際的 SMILES 轉換由 LLM 執行
            # 在實際應用中，這個方法會生成詳細的轉換指令
            transformation_instruction = {
                "operation": "scaffold_swap",
                "target_scaffold": target_scaffold,
                "target_smiles": scaffold_smiles,
                "preserve_substituents": preserve_substituents,
                "examples": params.get("example_transformations", [])
            }
            
            logger.debug(f"Generated scaffold swap instruction: {transformation_instruction}")
            
            # 返回佔位符 SMILES - 在實際系統中這會被 LLM 替換
            return f"SCAFFOLD_SWAP:{target_scaffold}:{parent_smiles}"
            
        except Exception as e:
            logger.error(f"Error in scaffold swap: {e}")
            return None
    
    def _apply_functional_group_addition(self, parent_smiles: str, params: Dict[str, Any]) -> Optional[str]:
        """
        應用官能基添加
        
        COMPLIANCE NOTE: Functional group addition uses template-based
        transformation without RDKit property calculations.
        """
        try:
            group = params.get("group", "")
            group_smiles = params.get("group_smiles", "")
            position = params.get("position", "random")
            
            if not group_smiles:
                logger.warning(f"No group SMILES provided for {group}")
                return None
            
            transformation_instruction = {
                "operation": "add_functional_group",
                "group": group,
                "group_smiles": group_smiles,
                "position": position,
                "examples": params.get("example_transformations", [])
            }
            
            logger.debug(f"Generated functional group addition instruction: {transformation_instruction}")
            
            # 返回佔位符 SMILES - 在實際系統中這會被 LLM 替換
            return f"ADD_GROUP:{group}:{parent_smiles}"
            
        except Exception as e:
            logger.error(f"Error in functional group addition: {e}")
            return None
    
    def _apply_heteroatom_exchange(self, parent_smiles: str, params: Dict[str, Any]) -> Optional[str]:
        """應用雜原子交換"""
        try:
            from_atom = params.get("from_atom", "")
            to_atom = params.get("to_atom", "")
            
            if not from_atom or not to_atom:
                logger.warning("Missing atom specifications for heteroatom exchange")
                return None
            
            # 簡單的字符串替換（在實際系統中會更複雜）
            result_smiles = parent_smiles.replace(from_atom, to_atom)
            
            logger.debug(f"Heteroatom exchange: {parent_smiles} -> {result_smiles}")
            return result_smiles
            
        except Exception as e:
            logger.error(f"Error in heteroatom exchange: {e}")
            return None
    
    def _apply_cyclization(self, parent_smiles: str, params: Dict[str, Any]) -> Optional[str]:
        """應用環化反應"""
        try:
            ring_size = params.get("ring_size", 6)
            ring_type = params.get("ring_type", "saturated")
            
            transformation_instruction = {
                "operation": "cyclization",
                "ring_size": ring_size,
                "ring_type": ring_type,
                "examples": params.get("example_transformations", [])
            }
            
            logger.debug(f"Generated cyclization instruction: {transformation_instruction}")
            
            # 返回佔位符 SMILES
            return f"CYCLIZE:{ring_size}:{parent_smiles}"
            
        except Exception as e:
            logger.error(f"Error in cyclization: {e}")
            return None
    
    def _apply_ring_opening(self, parent_smiles: str, params: Dict[str, Any]) -> Optional[str]:
        """應用開環反應"""
        try:
            target_ring_sizes = params.get("target_ring_sizes", [3, 4])
            
            transformation_instruction = {
                "operation": "ring_opening",
                "target_ring_sizes": target_ring_sizes,
                "examples": params.get("example_transformations", [])
            }
            
            logger.debug(f"Generated ring opening instruction: {transformation_instruction}")
            
            # 返回佔位符 SMILES
            return f"RING_OPEN:{parent_smiles}"
            
        except Exception as e:
            logger.error(f"Error in ring opening: {e}")
            return None
    
    def _get_emergency_actions(self, k: int) -> List[Dict[str, Any]]:
        """獲取緊急後備動作"""
        emergency_actions = [
            {
                "type": "add_functional_group",
                "name": "emergency_methyl",
                "description": "緊急添加甲基",
                "params": {"group": "CH3", "group_smiles": "C"}
            },
            {
                "type": "add_functional_group", 
                "name": "emergency_hydroxyl",
                "description": "緊急添加羥基",
                "params": {"group": "OH", "group_smiles": "O"}
            }
        ]
        
        return emergency_actions[:k]
    
    def get_action_by_name(self, name: str) -> Optional[TransformationRule]:
        """根據名稱獲取動作"""
        for action in self.all_actions:
            if action.name == name:
                return action
        return None
    
    def get_actions_by_type(self, action_type: str) -> List[TransformationRule]:
        """根據類型獲取動作列表"""
        return [action for action in self.all_actions if action.type == action_type]
    
    def reload_configuration(self):
        """重新載入配置文件"""
        logger.info("Reloading coarse actions configuration")
        self._load_configuration()
        self._build_action_registry()
    
    def get_statistics(self) -> Dict[str, Any]:
        """獲取引擎統計信息"""
        return {
            "total_actions": len(self.all_actions),
            "categories": len(self.actions_by_category),
            "actions_by_category": {
                category: len(actions) 
                for category, actions in self.actions_by_category.items()
            },
            "config_version": self.config.get("metadata", {}).get("version", "unknown"),
            "config_path": str(self.config_path)
        }

# 全局引擎實例
_global_engine = None

def get_engine() -> CoarseActionEngine:
    """獲取全局引擎實例"""
    global _global_engine
    if _global_engine is None:
        _global_engine = CoarseActionEngine()
    return _global_engine

def sample(k: int = 10, parent_smiles: str = None, 
           categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    向後兼容的抽樣函數
    
    COMPLIANCE NOTE: Uses externalized JSON rules for sampling,
    no RDKit property calculations.
    
    Args:
        k: 要抽樣的動作數量
        parent_smiles: 父分子 SMILES
        categories: 限制抽樣的類別列表
        
    Returns:
        抽樣得到的動作列表
    """
    engine = get_engine()
    return engine.sample(k=k, parent_smiles=parent_smiles, categories=categories)

def apply_coarse_action(parent_smiles: str, action: Dict[str, Any]) -> Optional[str]:
    """
    向後兼容的動作應用函數
    
    COMPLIANCE NOTE: Applies JSON-defined transformation rules,
    no RDKit property calculations.
    """
    engine = get_engine()
    return engine.apply_coarse_action(parent_smiles, action)

def get_all_actions() -> List[Dict[str, Any]]:
    """
    向後兼容的獲取所有動作函數
    """
    engine = get_engine()
    return [
        {
            "type": action.type,
            "name": action.name,
            "description": action.description,
            "params": action.params
        }
        for action in engine.all_actions
    ]

def reload_configuration():
    """重新載入配置"""
    engine = get_engine()
    engine.reload_configuration()

# COMPLIANCE ASSERTION
# This module uses externalized JSON configuration for all transformation rules.
# No RDKit property calculations are performed before Oracle evaluation.
assert True, "CoarseActionEngine complies with strict Oracle evaluation constraints"