from __future__ import print_function

import argparse
import heapq
import json
import os
import random
from time import time
from typing import List, Optional

import joblib
import numpy as np
from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.scoring_function import ScoringFunction
from guacamol.utils.chemistry import canonicalize
from guacamol.utils.helpers import setup_default_logger
from joblib import delayed
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
import csv

from .crossover import crossover
from .mutate import mutate


def make_mating_pool(population_mol: List[Mol], population_scores, offspring_size: int):
    """
    Given a population of RDKit Mol and their scores, sample a list of the same size
    with replacement using the population_scores as weights

    Args:
        population_mol: list of RDKit Mol
        population_scores: list of un-normalised scores given by ScoringFunction
        offspring_size: number of molecules to return

    Returns: a list of RDKit Mol (probably not unique)

    """
    # Handle case where scores are all zero or very low
    if all(score <= 0 for score in population_scores):
        # If all scores are zero or negative, use uniform distribution
        mating_pool = np.random.choice(population_mol, size=offspring_size, replace=True)
    else:
        # Convert negative scores to positive by adding offset
        min_score = min(population_scores)
        if min_score < 0:
            adjusted_scores = [score - min_score + 1e-8 for score in population_scores]
        else:
            adjusted_scores = [max(score, 1e-8) for score in population_scores]
        
        # scores -> probs
        sum_scores = sum(adjusted_scores)
        population_probs = [p / sum_scores for p in adjusted_scores]
        mating_pool = np.random.choice(population_mol, p=population_probs, size=offspring_size, replace=True)
    return mating_pool


def reproduce(mating_pool, mutation_rate):
    """

    Args:
        mating_pool: list of RDKit Mol
        mutation_rate: rate of mutation

    Returns:

    """
    parent_a = random.choice(mating_pool)
    parent_b = random.choice(mating_pool)
    new_child = crossover(parent_a, parent_b)
    if new_child is not None:
        new_child = mutate(new_child, mutation_rate)
    return new_child


def score_mol(mol, score_fn):
    return score_fn(Chem.MolToSmiles(mol))


def sanitize(population_mol):
    new_population = []
    smile_set = set()
    for mol in population_mol:
        if mol is not None:
            try:
                smile = Chem.MolToSmiles(mol)
                if smile is not None and smile not in smile_set:
                    smile_set.add(smile)
                    new_population.append(mol)
            except ValueError:
                print('bad smiles')
    return new_population



class SingleTaskWrapper:
    """Wrapper to run only a single benchmark task"""
    def __init__(self, generator, task_name):
        self.generator = generator
        self.task_name = task_name
        
    def generate_optimized_molecules(self, scoring_function, number_molecules, starting_population=None):
        return self.generator.generate_optimized_molecules(scoring_function, number_molecules, starting_population)



class SingleTaskWrapper:
    """Wrapper to run only a single benchmark task"""
    def __init__(self, generator, task_name):
        self.generator = generator
        self.task_name = task_name
        
    def generate_optimized_molecules(self, scoring_function, number_molecules, starting_population=None):
        return self.generator.generate_optimized_molecules(scoring_function, number_molecules, starting_population)


class GB_GA_Generator(GoalDirectedGenerator):

    def __init__(self, smi_file, population_size, offspring_size, generations, mutation_rate, n_jobs=-1, random_start=False, patience=5, task=None, output_dir=None):
        self.pool = joblib.Parallel(n_jobs=n_jobs)
        self.smi_file = smi_file
        self.all_smiles = self.load_smiles_from_file(self.smi_file)
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.random_start = random_start
        self.patience = patience
        self.task = task
        self.output_dir = output_dir

    def load_smiles_from_file(self, smi_file):
        with open(smi_file) as f:
            return self.pool(delayed(canonicalize)(s.strip()) for s in f)

    def top_k(self, smiles, scoring_function, k):
        joblist = (delayed(scoring_function.score)(s) for s in smiles)
        scores = self.pool(joblist)
        scored_smiles = list(zip(scores, smiles))
        scored_smiles = sorted(scored_smiles, key=lambda x: x[0], reverse=True)
        return [smile for score, smile in scored_smiles][:k]

    # --- replace old write_generation_csv with this ---
    def write_population_csv(self, generation, population_mol, population_scores):
        """
        Record the selected population (size = population_size) for each generation
        into results_population/{task}.csv
        """
        if self.task is None:
            return

        base_dir = self.output_dir or os.path.dirname(os.path.realpath(__file__))
        folder_path = os.path.join(base_dir, "results_population")
        os.makedirs(folder_path, exist_ok=True)
        filename = os.path.join(folder_path, f"{self.task}.csv")

        mode = "w" if generation == 0 and not os.path.isfile(filename) else "a"
        with open(filename, mode, newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["generation", "smiles", "score"])
            for mol, score in zip(population_mol, population_scores):
                writer.writerow([generation, Chem.MolToSmiles(mol), score])

    # --- new helper for offspring ---
    def write_offspring_csv(self, generation, offspring_mol, offspring_scores):
        """
        Record every offspring molecule generated this generation
        into results_offspring/{task}.csv
        """
        if self.task is None:
            return

        base_dir = self.output_dir or os.path.dirname(os.path.realpath(__file__))
        folder_path = os.path.join(base_dir, "results_offspring")
        os.makedirs(folder_path, exist_ok=True)
        filename = os.path.join(folder_path, f"{self.task}.csv")

        mode = "w" if generation == 0 and not os.path.isfile(filename) else "a"
        with open(filename, mode, newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["generation", "smiles", "score"])
            for mol, score in zip(offspring_mol, offspring_scores):
                if mol is None:
                    continue
                writer.writerow([generation, Chem.MolToSmiles(mol), score])

    def generate_optimized_molecules(self, scoring_function: ScoringFunction, number_molecules: int,
                                     starting_population: Optional[List[str]] = None) -> List[str]:

        if number_molecules > self.population_size:
            self.population_size = number_molecules
            print(f'Benchmark requested more molecules than expected: new population is {number_molecules}')

        # fetch initial population?
        if starting_population is None:
            print('selecting initial population...')
            if self.random_start:
                starting_population = np.random.choice(self.all_smiles, self.population_size)
            else:
                # ALWAYS use scoring_function to find lowest-scoring molecules as initial population for each task
                print('Finding lowest-scoring molecules for initial population...')
                joblist = (delayed(scoring_function.score)(s) for s in self.all_smiles)
                scores = self.pool(joblist)
                scored_smiles = list(zip(scores, self.all_smiles))
                scored_smiles = sorted(scored_smiles, key=lambda x: x[0])  # Sort by score ascendingly (lowest first)
                starting_population = [smile for score, smile in scored_smiles[:self.population_size]]

        # select initial population
        population_smiles = heapq.nlargest(self.population_size, starting_population, key=scoring_function.score)
        population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
        population_scores = self.pool(delayed(score_mol)(m, scoring_function.score) for m in population_mol)

        # evolution: go go go!!
        t0 = time()
        patience = 0

        for generation in range(self.generations):

            # new_population
            mating_pool = make_mating_pool(population_mol, population_scores, self.offspring_size)
            offspring_mol = self.pool(delayed(reproduce)(mating_pool, self.mutation_rate) for _ in range(self.offspring_size))

            # score and log offspring
            valid_offspring = [m for m in offspring_mol if m is not None]
            offspring_scores = self.pool(
                delayed(score_mol)(m, scoring_function.score) for m in valid_offspring
            )
            self.write_offspring_csv(generation, valid_offspring, offspring_scores)

            # add new_population
            population_mol += offspring_mol
            population_mol = sanitize(population_mol)

            # stats
            gen_time = time() - t0
            mol_sec = self.population_size / gen_time
            t0 = time()

            old_scores = population_scores
            population_scores = self.pool(delayed(score_mol)(m, scoring_function.score) for m in population_mol)
            population_tuples = list(zip(population_scores, population_mol))
            population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:self.population_size]
            population_mol = [t[1] for t in population_tuples]
            population_scores = [t[0] for t in population_tuples]

            # 每代結束時寫入 population csv
            self.write_population_csv(generation, population_mol, population_scores)

            # early stopping
            if population_scores == old_scores:
                patience += 1
                print(f'Failed to progress: {patience}')
                if patience >= self.patience:
                    print(f'No more patience, bailing...')
                    break
            else:
                patience = 0

            print(f'{generation} | '
                  f'max: {np.max(population_scores):.3f} | '
                  f'avg: {np.mean(population_scores):.3f} | '
                  f'min: {np.min(population_scores):.3f} | '
                  f'std: {np.std(population_scores):.3f} | '
                  f'sum: {np.sum(population_scores):.3f} | '
                  f'{gen_time:.2f} sec/gen | '
                  f'{mol_sec:.2f} mol/sec')

        # finally
        return [Chem.MolToSmiles(m) for m in population_mol][:number_molecules]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles_file', default='data/guacamol_v1_all.smiles')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--population_size', type=int, default=15)
    parser.add_argument('--offspring_size', type=int, default=30)
    parser.add_argument('--mutation_rate', type=float, default=0.5)
    parser.add_argument('--generations', type=int, default=1000)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--random_start', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--suite', default='v2')
    parser.add_argument('--task', type=str, required=True, help='任務名稱，決定csv檔名')

    args = parser.parse_args()

    np.random.seed(args.seed)

    setup_default_logger()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.realpath(__file__))

    # save command line args
    with open(os.path.join(args.output_dir, 'goal_directed_params.json'), 'w') as jf:
        json.dump(vars(args), jf, sort_keys=True, indent=4)

    optimiser = GB_GA_Generator(smi_file=args.smiles_file,
                                population_size=args.population_size,
                                offspring_size=args.offspring_size,
                                generations=args.generations,
                                mutation_rate=args.mutation_rate,
                                n_jobs=args.n_jobs,
                                random_start=args.random_start,
                                patience=args.patience,
                                task=args.task,
                                output_dir=args.output_dir)

    json_file_path = os.path.join(args.output_dir, f'{args.task}_results.json')
    
    import guacamol.assess_goal_directed_generation
    
    # Monkey-patch to filter benchmarks
    original_evaluate = guacamol.assess_goal_directed_generation._evaluate_goal_directed_benchmarks
    
    task_to_benchmark = {
        'osimertinib': 'Osimertinib MPO',
        'fexofenadine': 'Fexofenadine MPO',
        'ranolazine': 'Ranolazine MPO',
        'amlodipine': 'Amlodipine MPO',
        'perindopril': 'Perindopril MPO',
        'sitagliptin': 'Sitagliptin MPO',
        'zaleplon': 'Zaleplon MPO',
        'cobimetinib': 'Scaffold Hop',
        'scaffold_hop': 'Scaffold Hop',
        'decoration_hop': 'Decoration Hop',
        'weird_physchem': 'Weird physchem',
        'valsartan_smarts': 'Valsartan SMARTS',
        'median1': 'Median molecules 1',
        'median2': 'Median molecules 2',
        'isomer_c11h24': 'C11H24',
        'isomer_c9h10n2o2pf2cl': 'C9H10N2O2PF2Cl',
        'celecoxib': 'Celecoxib Rediscovery',
        'troglitazone': 'Troglitazone',
        'thiothixene': 'Thiothixene',
        'mestranol': 'Mestranol'

    }
    
    target_benchmark_name = task_to_benchmark.get(args.task, args.task)
    
    def filtered_evaluate(goal_directed_molecule_generator, benchmarks):
        filtered_benchmarks = []
        for benchmark in benchmarks:
            if benchmark.name == target_benchmark_name:
                filtered_benchmarks.append(benchmark)
                print(f"Running single benchmark: {benchmark.name}")
                break
        
        if not filtered_benchmarks:
            print(f"Error: No benchmark found for task '{args.task}' (mapped to '{target_benchmark_name}')")
            print("Available benchmarks:")
            for b in benchmarks:
                print(f"  - {b.name}")
            exit(1)
            
        return original_evaluate(goal_directed_molecule_generator, filtered_benchmarks)
    
    guacamol.assess_goal_directed_generation._evaluate_goal_directed_benchmarks = filtered_evaluate
    
    assess_goal_directed_generation(optimiser, json_output_file=json_file_path, benchmark_version=args.suite)


if __name__ == "__main__":
    main()