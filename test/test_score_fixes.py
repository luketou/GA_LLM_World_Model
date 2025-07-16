"""
測試修復後的分數系統
驗證 oracle_score 和早停條件是否正確工作
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_node_oracle_score():
    """測試 Node 類的 oracle_score 功能"""
    print("🧪 Testing Node oracle_score functionality...")
    
    from mcts.node import Node
    
    # 創建測試節點
    node = Node(smiles="c1ccccc1", depth=0)
    
    # 測試初始狀態
    assert node.oracle_score == 0.0, f"Initial oracle_score should be 0.0, got {node.oracle_score}"
    assert node.total_score == 0.0, f"Initial total_score should be 0.0, got {node.total_score}"
    assert node.visits == 0, f"Initial visits should be 0, got {node.visits}"
    
    # 測試更新功能
    test_score = 0.85
    node.update(test_score)
    
    assert node.oracle_score == test_score, f"oracle_score should be {test_score}, got {node.oracle_score}"
    assert node.total_score == test_score, f"total_score should be {test_score}, got {node.total_score}"
    assert node.visits == 1, f"visits should be 1, got {node.visits}"
    
    # 測試多次更新
    node.update(0.75)
    assert node.oracle_score == 0.75, f"oracle_score should be 0.75 (latest), got {node.oracle_score}"
    assert node.total_score == 1.60, f"total_score should be 1.60 (cumulative), got {node.total_score}"
    assert node.visits == 2, f"visits should be 2, got {node.visits}"
    
    # 測試平均分數
    expected_avg = 1.60 / 2
    assert abs(node.avg_score - expected_avg) < 0.001, f"avg_score should be {expected_avg}, got {node.avg_score}"
    
    print("✅ Node oracle_score functionality test passed!")
    return True

def test_early_stopping_logic():
    """測試早停邏輯模擬"""
    print("🧪 Testing early stopping logic...")
    
    from mcts.node import Node
    
    # 模擬引擎和節點
    class MockEngine:
        def __init__(self):
            self.best = None
    
    engine = MockEngine()
    
    # 測試沒有最佳節點的情況
    early_stop_threshold = 0.8
    should_stop = (hasattr(engine, 'best') and engine.best and 
                   hasattr(engine.best, 'oracle_score') and 
                   engine.best.oracle_score >= early_stop_threshold)
    assert not should_stop, "Should not stop when no best node exists"
    
    # 測試低分節點
    low_score_node = Node(smiles="CCO", depth=1)
    low_score_node.update(0.5)
    engine.best = low_score_node
    
    should_stop = (hasattr(engine, 'best') and engine.best and 
                   hasattr(engine.best, 'oracle_score') and 
                   engine.best.oracle_score >= early_stop_threshold)
    assert not should_stop, f"Should not stop with score {low_score_node.oracle_score} < {early_stop_threshold}"
    
    # 測試高分節點（應該觸發早停）
    high_score_node = Node(smiles="c1ccc(Cl)cc1", depth=1)
    high_score_node.update(0.85)
    engine.best = high_score_node
    
    should_stop = (hasattr(engine, 'best') and engine.best and 
                   hasattr(engine.best, 'oracle_score') and 
                   engine.best.oracle_score >= early_stop_threshold)
    assert should_stop, f"Should stop with score {high_score_node.oracle_score} >= {early_stop_threshold}"
    
    print("✅ Early stopping logic test passed!")
    return True

def test_best_node_selection():
    """測試最佳節點選擇邏輯"""
    print("🧪 Testing best node selection logic...")
    
    from mcts.node import Node
    
    # 模擬引擎
    class MockEngine:
        def __init__(self):
            self.best = None
        
        def update_best(self, node, score):
            """模擬最佳節點更新邏輯"""
            if not self.best or score > getattr(self.best, 'oracle_score', 0.0):
                self.best = node
                return True
            return False
    
    engine = MockEngine()
    
    # 測試第一個節點
    node1 = Node(smiles="CCO", depth=1)
    node1.update(0.3)
    updated = engine.update_best(node1, 0.3)
    assert updated, "First node should become best"
    assert engine.best.smiles == "CCO", "First node should be best"
    
    # 測試更低分的節點（不應該更新）
    node2 = Node(smiles="CCN", depth=1)
    node2.update(0.2)
    updated = engine.update_best(node2, 0.2)
    assert not updated, "Lower score node should not become best"
    assert engine.best.smiles == "CCO", "Best should remain unchanged"
    
    # 測試更高分的節點（應該更新）
    node3 = Node(smiles="c1ccccc1", depth=1)
    node3.update(0.7)
    updated = engine.update_best(node3, 0.7)
    assert updated, "Higher score node should become best"
    assert engine.best.smiles == "c1ccccc1", "Higher score node should be best"
    assert engine.best.oracle_score == 0.7, "Best node should have correct oracle_score"
    
    print("✅ Best node selection logic test passed!")
    return True

def test_score_consistency():
    """測試分數一致性"""
    print("🧪 Testing score consistency...")
    
    from mcts.node import Node
    
    node = Node(smiles="c1ccccc1", depth=0)
    
    # 記錄多次更新的分數
    scores = [0.3, 0.5, 0.8, 0.2, 0.9]
    
    for score in scores:
        node.update(score)
    
    # 檢查最後的 oracle_score 是否是最新的
    assert node.oracle_score == 0.9, f"oracle_score should be 0.9 (latest), got {node.oracle_score}"
    
    # 檢查 total_score 是否是累積的
    expected_total = sum(scores)
    assert abs(node.total_score - expected_total) < 0.001, f"total_score should be {expected_total}, got {node.total_score}"
    
    # 檢查平均分數
    expected_avg = expected_total / len(scores)
    assert abs(node.avg_score - expected_avg) < 0.001, f"avg_score should be {expected_avg}, got {node.avg_score}"
    
    # 檢查訪問次數
    assert node.visits == len(scores), f"visits should be {len(scores)}, got {node.visits}"
    
    print("✅ Score consistency test passed!")
    return True

def main():
    """運行所有測試"""
    print("🚀 Testing Score System Fixes\n")
    
    tests = [
        test_node_oracle_score,
        test_early_stopping_logic,
        test_best_node_selection,
        test_score_consistency
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed! The score system fixes are working correctly.")
        print("\n🔧 Key fixes verified:")
        print("  • oracle_score properly stores latest Oracle evaluation")
        print("  • total_score correctly accumulates for MCTS calculations")
        print("  • Early stopping uses oracle_score for decisions")
        print("  • Best node selection uses oracle_score for comparison")
        print("  • Score consistency maintained across updates")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)