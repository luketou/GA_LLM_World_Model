"""
Test script for LLM-guided action selection
Tests the new trajectory-aware molecular editing functionality
"""
import sys
import os
import asyncio
import logging

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_node_action_history():
    """Test the enhanced Node class with action history tracking"""
    print("Testing Node action history functionality...")
    
    from mcts.node import Node
    
    # Create root node
    root = Node(smiles="CC", depth=0)
    
    # Create child with action history
    action1 = {
        "type": "substitute",
        "name": "add_hydroxyl", 
        "description": "添加羥基",
        "params": {"smiles": "O"}
    }
    
    child1 = root.add_child("CCO", generating_action=action1)
    child1.update(0.5)  # Simulate score update
    
    # Create grandchild
    action2 = {
        "type": "substitute",
        "name": "add_methyl",
        "description": "添加甲基", 
        "params": {"smiles": "C"}
    }
    
    grandchild = child1.add_child("CCOC", generating_action=action2)
    grandchild.update(0.7)
    
    # Test action history retrieval
    history = grandchild.get_action_history()
    print(f"Action history length: {len(history)}")
    for i, record in enumerate(history):
        print(f"  {i+1}. {record['action']['name']} -> {record['smiles'][:20]}...")
    
    # Test trajectory summary
    summary = grandchild.get_action_trajectory_summary()
    print(f"Trajectory summary: {summary}")
    
    # Test successful patterns
    patterns = grandchild.get_successful_action_patterns()
    print(f"Successful patterns: {len(patterns)}")
    
    print("✓ Node action history test passed\n")
    return True

def test_llm_guided_selector():
    """Test the LLM-guided action selector (mock test)"""
    print("Testing LLM-guided action selector...")
    
    try:
        from mcts.llm_guided_selector import LLMGuidedActionSelector, ActionSelectionRequest
        
        # Mock LLM generator
        class MockLLMGenerator:
            def generate_text(self, prompt):
                return '''
                {
                  "selected_action_names": ["add_hydroxyl", "add_methyl"],
                  "reasoning": "Based on the trajectory showing successful polar group additions, continuing with hydroxyl group would build on the established pattern of improving solubility. Methyl addition provides a balance for lipophilicity.",
                  "confidence": 0.8
                }
                '''
        
        # Create selector
        llm_gen = MockLLMGenerator()
        selector = LLMGuidedActionSelector(llm_gen)
        
        # Create mock trajectory
        trajectory = {
            'total_actions': 2,
            'action_types': ['substitute'],
            'action_type_counts': {'substitute': 2},
            'score_trend': 'improving',
            'recent_actions': [
                {
                    'action': {'name': 'add_hydroxyl', 'type': 'substitute'},
                    'score_improvement': 0.2
                }
            ],
            'current_depth': 2,
            'avg_score': 0.6
        }
        
        # Create mock available actions
        available_actions = [
            {"name": "add_hydroxyl", "type": "substitute", "description": "添加羥基"},
            {"name": "add_methyl", "type": "substitute", "description": "添加甲基"},
            {"name": "add_amino", "type": "substitute", "description": "添加氨基"}
        ]
        
        # Create selection request
        request = ActionSelectionRequest(
            parent_smiles="CCO",
            current_node_trajectory=trajectory,
            available_actions=available_actions,
            optimization_goal="Improve drug-like properties",
            depth=2,
            max_selections=2
        )
        
        # Test selection
        response = selector.select_actions(request)
        
        print(f"Selected {len(response.selected_actions)} actions")
        print(f"Reasoning: {response.reasoning[:100]}...")
        print(f"Confidence: {response.confidence}")
        print(f"Fallback used: {response.fallback_used}")
        
        # Verify results
        assert len(response.selected_actions) > 0, "No actions selected"
        assert response.reasoning, "No reasoning provided"
        assert 0 <= response.confidence <= 1, "Invalid confidence value"
        
        print("✓ LLM-guided selector test passed\n")
        return True
        
    except Exception as e:
        print(f"✗ LLM-guided selector test failed: {e}")
        return False

def test_integration():
    """Test integration with MCTS engine (mock test)"""
    print("Testing MCTS engine integration...")
    
    try:
        # Mock the required modules
        class MockKGStore:
            pass
        
        class MockLLMGenerator:
            def generate_text(self, prompt):
                return '{"selected_action_names": ["add_methyl"], "reasoning": "Test reasoning", "confidence": 0.7}'
            
            def generate_batch(self, parent_smiles, actions):
                return [f"{parent_smiles}_modified_{i}" for i in range(len(actions))]
        
        from mcts.mcts_engine import MCTSEngine
        
        # Create engine with mock components
        kg_store = MockKGStore()
        llm_gen = MockLLMGenerator()
        engine = MCTSEngine(kg_store, max_depth=3, llm_gen=llm_gen)
        
        # Test that LLM-guided selector was initialized
        assert engine.llm_guided_selector is not None, "LLM-guided selector not initialized"
        
        # Test action proposal (this should use fallback since we don't have real actions modules)
        actions = engine.propose_actions("CC", depth=1, k_init=5)
        
        print(f"Proposed {len(actions)} actions")
        print(f"LLM-guided selector available: {engine.llm_guided_selector is not None}")
        
        assert len(actions) > 0, "No actions proposed"
        
        print("✓ MCTS engine integration test passed\n")
        return True
        
    except Exception as e:
        print(f"✗ MCTS engine integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Testing LLM-Guided Action Selection Implementation\n")
    
    tests = [
        test_node_action_history,
        test_llm_guided_selector, 
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! LLM-guided action selection is ready.")
        print("\n🚀 Key Features Implemented:")
        print("  • Node action history tracking")
        print("  • Trajectory-aware LLM prompt generation")
        print("  • Intelligent action selection with reasoning")
        print("  • Fallback mechanisms for robust operation")
        print("  • Integration with existing MCTS workflow")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)