import asyncio, yaml, pathlib
from graph.workflow_graph import graph_app, AgentState, oracle, cfg

def prescan():
    seed, score = oracle.prescan_lowest(cfg["smi_file"])
    print(f"[Prescan] Lowest-score seed: {seed} ({score:.4f})")
    return seed

async def run():
    seed = prescan()
    init_state = AgentState(parent_smiles=seed, depth=0)
    async for st in graph_app.astream(init_state):
        if st.result:
            best = st.result["best"]
            print(f"BEST: {best.smiles}  score={best.mean_score:.3f}")
            break

if __name__ == "__main__":
    asyncio.run(run())