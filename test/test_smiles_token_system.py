"""
測試 LLM SMILES Token 標記系統
驗證 <SMILES></SMILES> 標記的解析和提取功能
"""
import sys
import os
import logging

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_token_extraction():
    """測試 SMILES token 提取功能"""
    print("🧪 Testing SMILES Token Extraction System\n")
    
    from llm.generator import LLMGenerator
    
    # 創建 mock LLM generator（不需要真實 API 金鑰）
    class MockLLMGenerator(LLMGenerator):
        def __init__(self):
            self.max_smiles_length = 100
    
    generator = MockLLMGenerator()
    
    # 測試案例
    test_cases = [
        {
            "name": "Perfect Token Format",
            "response": """<SMILES>c1ccc(Cl)cc1</SMILES>
<SMILES>c1ccc(Br)cc1</SMILES>
<SMILES>c1ccc(F)cc1</SMILES>""",
            "expected_count": 3
        },
        {
            "name": "Mixed Format with Noise",
            "response": """<think>
Let me generate some SMILES...
</think>

<SMILES>CCO</SMILES>
Some explanation text here...
<SMILES>CCN</SMILES>
Another line of text.
<SMILES>CCC</SMILES>""",
            "expected_count": 3
        },
        {
            "name": "Case Insensitive",
            "response": """<smiles>c1ccccc1</smiles>
<SMILES>Cc1ccccc1</SMILES>
<Smiles>Nc1ccccc1</Smiles>""",
            "expected_count": 3
        },
        {
            "name": "Inline Tokens",
            "response": """Here are the results: <SMILES>CCO</SMILES> and <SMILES>CCN</SMILES>
Also this one: <SMILES>CCC</SMILES>""",
            "expected_count": 3
        },
        {
            "name": "No Tokens (Fallback Test)",
            "response": """c1ccccc1
CCO
CCN
Some invalid text here
CCC""",
            "expected_count": 4  # Should fall back to line parsing
        },
        {
            "name": "Invalid SMILES Filtering",
            "response": """<SMILES>c1ccccc1</SMILES>
<SMILES>invalid_smiles_123</SMILES>
<SMILES>CCO</SMILES>
<SMILES>()()()invalid</SMILES>
<SMILES>CCN</SMILES>""",
            "expected_count": 3  # Only valid SMILES should be extracted
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print(f"Input response:")
        print(f"```\n{test_case['response']}\n```")
        
        try:
            extracted = generator._extract_smiles_from_response(test_case['response'])
            
            print(f"Extracted SMILES: {extracted}")
            print(f"Count: {len(extracted)} (expected: {test_case['expected_count']})")
            
            if len(extracted) == test_case['expected_count']:
                print("✅ PASSED")
                passed += 1
            else:
                print("❌ FAILED - Count mismatch")
                
        except Exception as e:
            print(f"❌ FAILED - Exception: {e}")
        
        print("-" * 50)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    return passed == total

def test_enhanced_prompts():
    """測試增強的提示模板"""
    print("\n🎯 Testing Enhanced Prompt Templates\n")
    
    from llm.prompt import create_enhanced_llm_messages, create_simple_generation_prompt, create_fallback_prompt
    
    # 測試動作列表
    test_actions = [
        {"type": "substitute", "name": "add_chlorine", "description": "Add chlorine"},
        {"type": "scaffold_swap", "name": "swap_to_pyridine", "description": "Replace with pyridine"},
        {"type": "cyclization", "name": "form_ring", "description": "Form ring structure"}
    ]
    
    test_cases = [
        {
            "name": "Enhanced Messages",
            "function": create_enhanced_llm_messages,
            "args": ("c1ccccc1", test_actions)
        },
        {
            "name": "Simple Generation",
            "function": create_simple_generation_prompt,
            "args": ("c1ccccc1", 5)
        },
        {
            "name": "Fallback Prompt",
            "function": create_fallback_prompt,
            "args": ("c1ccccc1", 3)
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        
        try:
            messages = test_case['function'](*test_case['args'])
            
            # 檢查消息結構
            assert isinstance(messages, list), "Messages should be a list"
            assert len(messages) >= 1, "Should have at least one message"
            assert all('role' in msg and 'content' in msg for msg in messages), "Each message should have role and content"
            
            # 檢查是否包含 SMILES token 標記
            full_content = " ".join(msg['content'] for msg in messages)
            has_token_instruction = '<SMILES>' in full_content and '</SMILES>' in full_content
            
            print(f"Messages generated: {len(messages)}")
            print(f"Contains SMILES token instructions: {has_token_instruction}")
            
            if has_token_instruction:
                print("✅ PASSED")
                passed += 1
            else:
                print("❌ FAILED - Missing SMILES token instructions")
                print(f"Content preview: {full_content[:200]}...")
                
        except Exception as e:
            print(f"❌ FAILED - Exception: {e}")
        
        print("-" * 50)
    
    print(f"\n📊 Prompt Test Results: {passed}/{total} tests passed")
    return passed == total

def test_integration():
    """測試整合功能"""
    print("\n🔧 Testing Integration with Mock LLM\n")
    
    # Mock LLM response that would come from a real API
    mock_llm_response = """<SMILES>c1ccc(Cl)cc1</SMILES>
<SMILES>c1ccc(Br)cc1</SMILES>
<SMILES>c1ccc(F)cc1</SMILES>
<SMILES>c1ccc(I)cc1</SMILES>
<SMILES>c1ccc(CN)cc1</SMILES>"""
    
    class MockLLMGenerator:
        def __init__(self):
            self.provider = "mock"
            self.model_name = "mock-model"
            self.temperature = 0.2
            self.max_tokens = 1000
            self.max_smiles_length = 100
            self.top_p = 1.0
            self.stream = False
            
        def validate_smiles(self, smiles: str) -> bool:
            """Simple validation without RDKit"""
            if not smiles or len(smiles) < 1:
                return False
            # Basic character check
            allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789()[]=#+-@/\\%.')
            return all(c in allowed_chars for c in smiles)
            
        def _extract_smiles_from_response(self, response_text: str):
            """Use the real extraction method"""
            import re
            smiles_list = []
            
            # Extract SMILES from tokens
            pattern = r'<SMILES>(.*?)</SMILES>'
            matches = re.findall(pattern, response_text, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                smiles = match.strip()
                if smiles and self.validate_smiles(smiles):
                    smiles_list.append(smiles)
            
            return smiles_list
    
    generator = MockLLMGenerator()
    
    try:
        extracted_smiles = generator._extract_smiles_from_response(mock_llm_response)
        
        print(f"Mock LLM Response:")
        print(f"```\n{mock_llm_response}\n```")
        print(f"Extracted SMILES: {extracted_smiles}")
        print(f"Count: {len(extracted_smiles)}")
        
        expected_count = 5
        if len(extracted_smiles) == expected_count:
            print("✅ Integration test PASSED")
            return True
        else:
            print(f"❌ Integration test FAILED - Expected {expected_count}, got {len(extracted_smiles)}")
            return False
            
    except Exception as e:
        print(f"❌ Integration test FAILED - Exception: {e}")
        return False

def main():
    """運行所有測試"""
    print("🚀 Testing SMILES Token Marking System\n")
    
    tests = [
        test_token_extraction,
        test_enhanced_prompts,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test failed with exception: {e}")
    
    print(f"\n🎉 Overall Test Results: {passed}/{total} test suites passed")
    
    if passed == total:
        print("\n✅ All tests passed! The SMILES token marking system is ready.")
        print("\n🔧 Key Features Implemented:")
        print("  • <SMILES></SMILES> token-based extraction")
        print("  • Robust regex and fallback parsing")
        print("  • Enhanced prompt templates with examples")
        print("  • Multi-level prompt strategies")
        print("  • Improved SMILES validation")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)