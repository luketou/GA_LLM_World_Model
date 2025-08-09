#!/bin/bash
# Wrapper script to run llm_muvera_rl_critic.py without GLIBC conflicts

# Save current LD_LIBRARY_PATH
OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH

# Unset LD_LIBRARY_PATH to avoid GLIBC conflicts
unset LD_LIBRARY_PATH

# Run the Python script with all arguments passed to this wrapper
python /home/luketou/GA_LLM_World_Model/llm_teach/llm_muvera_rl_critic.py "$@"

# Restore LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$OLD_LD_LIBRARY_PATH
