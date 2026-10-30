"""Verify paired run provenance and exact checkpoint-to-sample reproduction."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import DEFAULT_CONFIG, config_from_dict
from data.artifacts import load_bundle, sha256
from data.normalization import NormalizationStats
from diffusion.noise_schedule import NoiseSchedule
from diffusion.sampling import sample
from evaluation.metrics import evaluate_samples
from experiments.compare import verify_pairing, CONFIGURATIONS
from models.factory import make_model
from training.train import restore_checkpoint
from utils.reproducibility import save_json


def verify_runs(results: Path, run_id: str, data: Path) -> dict:
    records = [json.loads((results/f'{run_id}_{name}.json').read_text()) for name in CONFIGURATIONS]
    verify_pairing(records)
    bundle = load_bundle(data, DEFAULT_CONFIG)
    reports = {}
    for record in records:
        name = record['model_choice']
        checkpoint, samples_path = Path(record['checkpoint']), Path(record['samples_file'])
        if record['dataset_sha256'] != bundle.metadata['sha256']:
            raise ValueError('Dataset provenance mismatch')
        if sha256(checkpoint) != record['checkpoint_sha256'] or sha256(samples_path) != record['samples_sha256']:
            raise ValueError('Checkpoint/sample checksum mismatch')
        config = config_from_dict(record['config'])
        model = make_model(name,record['seed'],config)
        schedule = NoiseSchedule(config.diffusion)
        payload = restore_checkpoint(model,schedule,checkpoint)
        if json.loads(json.dumps(payload['config'])) != record['config'] or payload['normalization'] != record['normalization']:
            raise ValueError('Checkpoint configuration/normalization mismatch')
        reproduced = sample(model,schedule,record['sampling']['n_sequences'],
                            record['sampling']['seed'],record['config']['training']['batch_size'])
        physical = NormalizationStats(**record['normalization']).inverse_transform(reproduced.samples.numpy())
        if not np.array_equal(physical,np.load(samples_path,allow_pickle=False)):
            raise ValueError(f'Checkpoint sample reproduction failed: {name}')
        metrics = evaluate_samples(physical,bundle.dataset.sequences[bundle.splits['test']],
                                   record['config']['evaluation']['max_acf_lag'])
        if metrics != record['generation_quality_metrics']:
            raise ValueError(f'Held-out metric reproduction failed: {name}')
        if name == 'hybrid_acf':
            for component in ('encoder','decoder','timestep_encoding','bottleneck'):
                if not all(row[component] > 0 for row in record['training']['gradient_norms_per_update']):
                    raise ValueError(f'Missing guided training gradient: {component}')
                if record['training']['last_model_parameter_change_l2'][component] <= 0:
                    raise ValueError(f'No guided parameter update: {component}')
        reports[name] = {'checkpoint_checksum':'passed','sample_checksum':'passed',
                         'checkpoint_sample_reproduction':'exact','held_out_metric_reproduction':'exact',
                         'generated_sequences':len(physical)}
    return {'status':'passed','pairing':'passed','runs':reports,
            'guided_joint_gradient_and_update_checks':'passed',
            'scope':'Reproducibility checks alone do not establish convergence or quantum advantage'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--results-dir',type=Path,default=ROOT/'results')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data/datasets')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Verification output exists; choose a fresh path')
    report = verify_runs(args.results_dir,args.run_id,args.data_dir)
    save_json(args.output,report)
    print(f'Three-configuration reproduction verified: {args.output}')


if __name__ == '__main__':
    main()
