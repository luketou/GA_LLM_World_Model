#!/bin/bash

# Script to run LLM selection on multiple tasks

TASKS=(
    osimertinib
    fexofenadine
    ranolazine
    amlodipine
    perindopril
    sitagliptin
    zaleplon
    celecoxib
    troglitazone
    thiothixene
)

# Create log directory if it doesn't exist
mkdir -p log

echo "Starting LLM selection for multiple tasks..."

for TASK in "${TASKS[@]}"
do
    # Check if the input CSV exists
    if [ -f "graph_ga/${TASK}.csv" ]; then
        echo "===== Processing task: $TASK =====" | tee -a log/llm_selection.log
        python llm_select.py --task "$TASK" 2>&1 | tee -a log/llm_selection.log
        echo "" | tee -a log/llm_selection.log
    else
        echo "Warning: graph_ga/${TASK}.csv not found, skipping $TASK" | tee -a log/llm_selection.log
    fi
done

echo "LLM selection completed!"
