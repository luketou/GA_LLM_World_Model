#!/usr/bin/env python3
"""
Test script for llm_select.py functionality without actually calling the LLM
"""

import csv
from collections import defaultdict

def test_load_csv():
    """Test loading and parsing CSV file"""
    csv_files = ['graph_ga/celecoxib.csv', 'graph_ga/osimertinib.csv']
    
    for csv_path in csv_files:
        try:
            generation_data = defaultdict(list)
            
            with open(csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    generation = int(row['generation'])
                    smiles = row['smiles']
                    score = float(row['score'])
                    generation_data[generation].append((smiles, score))
            
            print(f"\nFile: {csv_path}")
            print(f"Total generations: {len(generation_data)}")
            for gen in sorted(generation_data.keys())[:3]:  # Show first 3 generations
                mols = generation_data[gen]
                print(f"  Generation {gen}: {len(mols)} molecules")
                if mols:
                    print(f"    Top molecule: {mols[0][0][:50]}... (score: {mols[0][1]:.6f})")
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")

if __name__ == "__main__":
    test_load_csv()
