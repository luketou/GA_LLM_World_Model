"""Utilities for converting SMILES to model features."""
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_to_mol(smiles: str) -> Any:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol


def mol_to_fp(mol: Any) -> Any:
    # Morgan fingerprint as placeholder feature
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
