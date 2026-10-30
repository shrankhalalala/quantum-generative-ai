"""Run the teacher's three configurations with a single explicit common budget."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import DEFAULT_CONFIG
from utils.reproducibility import save_json

CONFIGURATIONS = ("classical", "hybrid", "hybrid_acf")


def verify_pairing(records: list[dict]) -> None:
    if [record['model_choice'] for record in records] != list(CONFIGURATIONS):
        raise ValueError('Expected classical, hybrid, hybrid_acf records in order')
    first = records[0]
    for record in records[1:]:
        for key in ('seed', 'subset_size', 'dataset_sha256', 'normalization', 'initial_shared_state_sha256'):
            if record[key] != first[key]:
                raise ValueError(f'Unpaired run field: {key}')
        for key in ('diffusion', 'training', 'evaluation'):
            if record['config'][key] != first['config'][key]:
                raise ValueError(f'Unpaired configuration: {key}')
        for key in ('steps', 'examples_seen', 'stream_seeds', 'checkpoint_selection'):
            if record['training'][key] != first['training'][key]:
                raise ValueError(f'Unpaired training: {key}')
        if record['sampling']['seed'] != first['sampling']['seed']:
            raise ValueError('Unpaired sampling seed')
    if records[1]['initial_model_state_sha256'] != records[2]['initial_model_state_sha256']:
        raise ValueError('Hybrid arms must start with identical full model states')


def comparison_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        metrics = record['generation_quality_metrics']
        rows.append({'configuration': record['model_choice'], 'seed': record['seed'],
                     'subset_size': record['subset_size'], 'updates': record['training']['steps'],
                     'lambda_acf': record['training']['effective_lambda_acf'],
                     'parameters': record['training']['trainable_parameters'],
                     'validation_noise_mse': record['training']['best_validation_loss'],
                     'wasserstein_1': metrics['marginal_wasserstein_1'], 'acf_mae': metrics['acf_mae'],
                     'mean_absolute_error': metrics['mean']['absolute_error'],
                     'std_absolute_error': metrics['std']['absolute_error'],
                     'variance_absolute_error': metrics['variance']['absolute_error'],
                     'training_seconds': record['training']['training_seconds'],
                     'sampling_seconds': record['sampling']['seconds']})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument('--smoke', action='store_true')
    budget.add_argument('--max-steps', type=int)
    budget.add_argument('--epochs', type=int)
    parser.add_argument('--lambda-acf', type=float, default=DEFAULT_CONFIG.loss.lambda_acf)
    parser.add_argument('--seed', type=int, choices=DEFAULT_CONFIG.training.seeds, default=11)
    parser.add_argument('--subset-size', type=int, choices=DEFAULT_CONFIG.dataset.subset_sizes, default=1000)
    parser.add_argument('--validation-every', type=int)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--results-dir', type=Path, default=ROOT/'results')
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data/datasets')
    args = parser.parse_args()
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):
        parser.error('Run ID must contain only letters, digits, underscore or hyphen')
    from dataclasses import replace
    replace(DEFAULT_CONFIG, loss=replace(DEFAULT_CONFIG.loss, lambda_acf=args.lambda_acf)).validate()
    for value in (args.max_steps, args.epochs, args.validation_every):
        if value is not None and value <= 0:
            parser.error('Budgets and validation interval must be positive')
    result_dir = args.results_dir.resolve()
    comparison = result_dir/f'{args.run_id}_comparison.json'
    table = result_dir/'tables'/f'{args.run_id}_comparison.csv'
    paths = [result_dir/f'{args.run_id}_{name}.json' for name in CONFIGURATIONS]
    checkpoints = [result_dir/'checkpoints'/f'{args.run_id}_{name}_best.pt' for name in CONFIGURATIONS]
    if any(path.exists() for path in [comparison, table, *paths, *checkpoints]):
        parser.error('Comparison artifacts already exist; choose a new run ID')
    budget_args = ['--smoke'] if args.smoke else (['--max-steps', str(args.max_steps)] if args.max_steps else ['--epochs', str(args.epochs)])
    for name in CONFIGURATIONS:
        command = [sys.executable, str(ROOT/'training/train.py'), '--model', name,
                   '--seed', str(args.seed), '--subset-size', str(args.subset_size),
                   '--run-id', f'{args.run_id}_{name}', '--results-dir', str(result_dir),
                   '--data-dir', str(args.data_dir.resolve()), *budget_args]
        if args.validation_every is not None:
            command += ['--validation-every', str(args.validation_every)]
        if name == 'hybrid_acf':
            command += ['--lambda-acf', str(args.lambda_acf)]
        subprocess.run(command, check=True, cwd=ROOT)
    records = [json.loads(path.read_text()) for path in paths]
    verify_pairing(records)
    rows = comparison_rows(records)
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(comparison, {'pairing': 'passed', 'pilot': args.smoke,
                          'lambda_status': 'User-specified; not automatically tuned',
                          'interpretation': 'Pilot diagnostics only' if args.smoke else 'Single-seed results; repeat before conclusions',
                          'runs': [str(path) for path in paths], 'rows': rows})
    print(f'Three-configuration comparison: {comparison}')


if __name__ == '__main__':
    main()
