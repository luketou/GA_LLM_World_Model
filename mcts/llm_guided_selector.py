"""
LLM-Guided Action Selector
Enhanced action selection using historical trajectory context for trajectory-aware molecular editing
"""
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ActionSelectionRequest:
    """Request structure for LLM-guided action selection"""
    parent_smiles: str
    current_node_trajectory: Dict[str, Any]
    available_actions: List[Dict[str, Any]]
    optimization_goal: str
    depth: int
    max_selections: int = 5

@dataclass 
class ActionSelectionResponse:
    """Response structure from LLM action selection"""
    selected_actions: List[Dict[str, Any]]
    reasoning: str
    confidence: float = 0.0
    fallback_used: bool = False

class LLMGuidedActionSelector:
    """
    LLM-guided action selector that uses historical trajectory context
    to make informed decisions about next molecular transformations
    """
    
    def __init__(self, llm_generator, max_context_actions: int = 5):
        """
        Initialize the LLM-guided action selector
        
        Args:
            llm_generator: LLM generator instance
            max_context_actions: Maximum number of historical actions to include in context
        """
        self.llm_gen = llm_generator
        self.max_context_actions = max_context_actions
        
    def select_actions(self, request: ActionSelectionRequest) -> ActionSelectionResponse:
        """
        Select the best actions using LLM guidance based on trajectory context
        
        Args:
            request: Action selection request with context
            
        Returns:
            ActionSelectionResponse with selected actions and reasoning
        """
        try:
            logger.info(f"LLM-guided action selection for {request.parent_smiles[:50]}...")
            
            # Prepare context for LLM
            context = self._prepare_llm_context(request)
            
            # Generate LLM prompt
            prompt = self._create_selection_prompt(context, request)
            
            # Query LLM for action selection
            llm_response = self._query_llm_for_selection(prompt)
            
            # Parse LLM response
            response = self._parse_llm_response(llm_response, request.available_actions)
            
            # Validate and filter selections
            validated_response = self._validate_selections(response, request)
            
            logger.info(f"LLM selected {len(validated_response.selected_actions)} actions with reasoning: {validated_response.reasoning[:100]}...")
            
            return validated_response
            
        except Exception as e:
            logger.error(f"Error in LLM-guided action selection: {e}")
            return self._fallback_selection(request)
    
    def _prepare_llm_context(self, request: ActionSelectionRequest) -> Dict[str, Any]:
        """Prepare comprehensive context for LLM"""
        trajectory = request.current_node_trajectory
        
        context = {
            "current_molecule": request.parent_smiles,
            "optimization_goal": request.optimization_goal,
            "current_depth": request.depth,
            "trajectory_summary": trajectory,
            "available_action_count": len(request.available_actions),
            "action_categories": self._categorize_available_actions(request.available_actions)
        }
        
        return context
    
    def _categorize_available_actions(self, actions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Categorize available actions by type"""
        categories = {}
        for action in actions:
            action_type = action.get('type', 'unknown')
            categories[action_type] = categories.get(action_type, 0) + 1
        return categories
    
    def _create_selection_prompt(self, context: Dict[str, Any], request: ActionSelectionRequest) -> str:
        """Create a comprehensive prompt for LLM action selection"""
        
        # Extract trajectory information
        trajectory = context["trajectory_summary"]
        recent_actions = trajectory.get("recent_actions", [])
        score_trend = trajectory.get("score_trend", "unknown")
        action_type_counts = trajectory.get("action_type_counts", {})
        
        # Build historical context
        history_context = ""
        if recent_actions:
            history_context = "Recent editing history:\n"
            for i, action_record in enumerate(recent_actions[-3:], 1):
                action = action_record.get('action', {})
                improvement = action_record.get('score_improvement', 0.0)
                history_context += f"  {i}. {action.get('name', 'Unknown')} ({action.get('type', 'unknown')}): {action.get('description', 'No description')} → Score change: {improvement:+.4f}\n"
        else:
            history_context = "This is the starting molecule with no previous modifications.\n"
        
        # Build action summary
        action_summary = ""
        if action_type_counts:
            action_summary = "Action types used so far: " + ", ".join([f"{atype}({count})" for atype, count in action_type_counts.items()]) + "\n"
        
        # Create available actions list
        available_actions_text = ""
        for i, action in enumerate(request.available_actions[:20], 1):  # Limit to first 20 to avoid token limits
            available_actions_text += f"  {i}. {action.get('name', 'Unknown')} ({action.get('type', 'unknown')}): {action.get('description', 'No description')}\n"
        
        if len(request.available_actions) > 20:
            available_actions_text += f"  ... and {len(request.available_actions) - 20} more actions\n"
        
        prompt = f"""You are an expert in molecular optimization and drug design. Your task is to select the most promising molecular transformation actions based on the editing trajectory and optimization goal.

CURRENT SITUATION:
- Current molecule: {request.parent_smiles}
- Optimization goal: {request.optimization_goal}
- Current search depth: {request.depth}
- Score trend: {score_trend}
- Current average score: {trajectory.get('avg_score', 0.0):.4f}

EDITING TRAJECTORY CONTEXT:
{history_context}
{action_summary}

AVAILABLE ACTIONS ({len(request.available_actions)} total):
{available_actions_text}

SELECTION CRITERIA:
1. Consider the trajectory context - what has worked well so far?
2. Look for complementary actions that build on successful patterns
3. Avoid repeating failed strategies (actions that decreased scores)
4. Balance exploration (trying new action types) with exploitation (using proven strategies)
5. Consider molecular diversity and avoid over-optimization

Please select the {min(request.max_selections, len(request.available_actions))} most promising actions and provide your reasoning.

RESPONSE FORMAT (JSON):
{{
  "selected_action_names": ["action_name_1", "action_name_2", ...],
  "reasoning": "Detailed explanation of why these actions were selected based on the trajectory context and optimization goal",
  "confidence": 0.8
}}

Your response:"""

        return prompt
    
    def _query_llm_for_selection(self, prompt: str) -> str:
        """Query LLM for action selection"""
        try:
            # Use the new generate_text_response method for general text generation
            response = self.llm_gen.generate_text_response(prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            raise
    
    def _parse_llm_response(self, llm_response: str, available_actions: List[Dict[str, Any]]) -> ActionSelectionResponse:
        """Parse LLM response and extract selected actions"""
        try:
            # Try to extract JSON from response
            response_text = llm_response.strip()
            
            # Find JSON block in response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in LLM response")
            
            json_text = response_text[start_idx:end_idx]
            response_data = json.loads(json_text)
            
            # Extract selected action names
            selected_names = response_data.get('selected_action_names', [])
            reasoning = response_data.get('reasoning', 'No reasoning provided')
            confidence = float(response_data.get('confidence', 0.5))
            
            # Map names to actual actions
            action_name_map = {action.get('name', ''): action for action in available_actions}
            selected_actions = []
            
            for name in selected_names:
                if name in action_name_map:
                    selected_actions.append(action_name_map[name])
                else:
                    # Try partial matching
                    for action_name, action in action_name_map.items():
                        if name.lower() in action_name.lower() or action_name.lower() in name.lower():
                            selected_actions.append(action)
                            break
            
            return ActionSelectionResponse(
                selected_actions=selected_actions,
                reasoning=reasoning,
                confidence=confidence,
                fallback_used=False
            )
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.debug(f"Raw LLM response: {llm_response}")
            
            # Try simple text parsing as fallback
            return self._simple_text_parse(llm_response, available_actions)
    
    def _simple_text_parse(self, response: str, available_actions: List[Dict[str, Any]]) -> ActionSelectionResponse:
        """Simple text parsing fallback for LLM response"""
        try:
            # Look for action names mentioned in the response
            mentioned_actions = []
            response_lower = response.lower()
            
            for action in available_actions:
                action_name = action.get('name', '').lower()
                if action_name and action_name in response_lower:
                    mentioned_actions.append(action)
            
            # If no actions found, use first few actions
            if not mentioned_actions:
                mentioned_actions = available_actions[:3]
            
            # Extract reasoning (look for explanation keywords)
            reasoning_start = max(
                response.lower().find('reasoning'),
                response.lower().find('because'),
                response.lower().find('explanation'),
                0
            )
            reasoning = response[reasoning_start:reasoning_start+200] if reasoning_start > 0 else "LLM selection based on text analysis"
            
            return ActionSelectionResponse(
                selected_actions=mentioned_actions[:5],
                reasoning=reasoning,
                confidence=0.3,
                fallback_used=True
            )
            
        except Exception as e:
            logger.error(f"Error in simple text parsing: {e}")
            raise
    
    def _validate_selections(self, response: ActionSelectionResponse, request: ActionSelectionRequest) -> ActionSelectionResponse:
        """Validate and ensure reasonable action selections"""
        # Ensure we don't exceed max selections
        if len(response.selected_actions) > request.max_selections:
            response.selected_actions = response.selected_actions[:request.max_selections]
        
        # Ensure we have at least one action
        if not response.selected_actions and request.available_actions:
            # Use first available action as fallback
            response.selected_actions = [request.available_actions[0]]
            response.reasoning += " [Fallback: Selected first available action]"
            response.fallback_used = True
        
        # Ensure diversity in action types if possible
        if len(response.selected_actions) > 1:
            response.selected_actions = self._ensure_action_diversity(response.selected_actions)
        
        return response
    
    def _ensure_action_diversity(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure diversity in selected action types"""
        if len(actions) <= 1:
            return actions
        
        diverse_actions = []
        seen_types = set()
        
        # First pass: one action per type
        for action in actions:
            action_type = action.get('type', 'unknown')
            if action_type not in seen_types:
                diverse_actions.append(action)
                seen_types.add(action_type)
        
        # Second pass: fill remaining slots
        remaining_slots = len(actions) - len(diverse_actions)
        for action in actions:
            if len(diverse_actions) >= len(actions):
                break
            if action not in diverse_actions:
                diverse_actions.append(action)
        
        return diverse_actions
    
    def _fallback_selection(self, request: ActionSelectionRequest) -> ActionSelectionResponse:
        """Fallback selection when LLM fails"""
        logger.warning("Using fallback action selection")
        
        # Simple heuristic-based selection
        selected_actions = []
        trajectory = request.current_node_trajectory
        
        # Prefer actions that have worked before
        successful_patterns = trajectory.get('recent_actions', [])
        successful_types = set()
        
        for action_record in successful_patterns:
            if action_record.get('score_improvement', 0.0) > 0:
                action_type = action_record.get('action', {}).get('type', '')
                if action_type:
                    successful_types.add(action_type)
        
        # Select actions based on successful patterns
        for action in request.available_actions:
            if len(selected_actions) >= request.max_selections:
                break
            
            action_type = action.get('type', '')
            if action_type in successful_types:
                selected_actions.append(action)
        
        # Fill remaining slots with diverse actions
        seen_types = {action.get('type', '') for action in selected_actions}
        for action in request.available_actions:
            if len(selected_actions) >= request.max_selections:
                break
            
            action_type = action.get('type', '')
            if action_type not in seen_types:
                selected_actions.append(action)
                seen_types.add(action_type)
        
        # If still need more, add any remaining actions
        for action in request.available_actions:
            if len(selected_actions) >= request.max_selections:
                break
            if action not in selected_actions:
                selected_actions.append(action)
        
        return ActionSelectionResponse(
            selected_actions=selected_actions[:request.max_selections],
            reasoning="Fallback selection: prioritized previously successful action types and ensured diversity",
            confidence=0.5,
            fallback_used=True
        )

# Factory function to create selector
def create_llm_guided_selector(llm_generator) -> LLMGuidedActionSelector:
    """Create an LLM-guided action selector instance"""
    return LLMGuidedActionSelector(llm_generator)