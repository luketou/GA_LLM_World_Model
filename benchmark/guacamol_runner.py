"""Run GuacaMol benchmark suite with the hybrid generator."""
import argparse
from typing import List

from guacamol.benchmark_suites import goal_directed_benchmark_suite
from guacamol.scoring_function import ScoringFunction

from agent.llm_agent import LangGraphAgent
from agent.graph_ga_wrapper import ga_generate
from agent.policy import MetaPolicy
from world_model import config, data_utils, encoder, heads, gate, task_encoder
from oracle import dft_proxy


def run_task(scoring_function: ScoringFunction, policy: MetaPolicy, task_name: str) -> float:
    llm = policy.llm_agent_generate
    gen_fn = policy.next_generator()
    smiles_list = gen_fn(5, None)
    scores = []
    for smi in smiles_list:
        mol = data_utils.smiles_to_mol(smi)
        fp = data_utils.mol_to_fp(mol)
        rep = encoder.MoleculeEncoder()(fp)
        props = heads.PropertyHeads(rep.shape[-1])(rep)
        novelty = rep.norm() * 0.0
        risk = gate.RiskGate()(props, novelty)
        if risk <= config.RISK_THRESHOLD:
            score = scoring_function.score(smi)
        else:
            score = 0.0
        policy.update(risk.item() * score)
        scores.append(score)
    return sum(scores) / len(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle_budget", type=int, default=200)
    args = parser.parse_args()
    suite = goal_directed_benchmark_suite()
    llm_agent = LangGraphAgent()
    policy = MetaPolicy(llm_agent.generate, ga_generate)
    results: List[float] = []
    for benchmark in suite:
        score = run_task(benchmark.scoring_function, policy, benchmark.name)
        results.append(score)
        print(f"{benchmark.name}: {score:.3f}")
    print(f"Average score: {sum(results)/len(results):.3f}")


if __name__ == "__main__":
    main()
