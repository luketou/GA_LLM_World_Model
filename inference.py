"""Inference script for a single SMILES."""
import argparse

from world_model import data_utils, encoder, heads, gate, task_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--op", required=True)
    parser.add_argument("--value", type=float, required=True)
    args = parser.parse_args()

    mol = data_utils.smiles_to_mol(args.smiles)
    fp = data_utils.mol_to_fp(mol)
    rep = encoder.MoleculeEncoder()(fp)
    props = heads.PropertyHeads(rep.shape[-1])(rep)
    novelty = rep.norm() * 0.0
    risk = gate.RiskGate()(props, novelty)
    print(f"Risk: {risk.item():.3f}")


if __name__ == "__main__":
    main()
