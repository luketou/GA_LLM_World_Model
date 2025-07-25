#!/usr/bin/env python3
"""
Example of how to use llm_select.py
"""

print("""
Example usage of llm_select.py:

1. Process a single task:
   python llm_select.py --task celecoxib

2. Process with custom parameters:
   python llm_select.py --task osimertinib --temperature 0.2 --max_tokens 512

3. Use different directories:
   python llm_select.py --task celecoxib --input_dir data/csv --output_dir results

4. Batch process multiple tasks:
   ./run_llm_selection.sh

The script will:
- Read molecules from graph_ga/{task}.csv
- Group molecules by generation
- For each generation with >10 molecules:
  - Ask LLM to select the 10 most promising ones
  - Consider task objectives, chemical feasibility, and diversity
- Save selected molecules to graph_ga/llm_{task}.csv

Output format matches input format:
- generation: generation number
- smiles: SMILES string of the molecule
- score: original score from the GA (not re-scored by LLM)
""")
