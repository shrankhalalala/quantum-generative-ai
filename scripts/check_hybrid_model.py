"""Verify saved Phase 5 pilots and estimate cost; never launch training."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from config import DEFAULT_CONFIG
from data.artifacts import load_bundle, sha256, subset_key
from data.normalization import NormalizationStats
from models.classical_denoiser import trainable_parameters
from models.factory import make_model, shared_state_sha256
from diffusion.noise_schedule import NoiseSchedule
from diffusion.sampling import sample
from training.train import restore_checkpoint
from utils.reproducibility import save_json

RUN_IDS={'classical':'phase5_classical_seed11_n1000','hybrid':'phase5_hybrid_seed11_n1000'}


def verify_pilots(results_dir: Path, data_dir: Path) -> dict:
    records={name:json.loads((results_dir/f'{run_id}.json').read_text()) for name,run_id in RUN_IDS.items()}
    baseline,hybrid=records['classical'],records['hybrid']
    for key in ('config','seed','subset_size','normalization','dataset_sha256','initial_shared_state_sha256'):
        if baseline[key]!=hybrid[key]:
            raise AssertionError(f'Paired pilot mismatch: {key}')
    for key in ('stream_seeds','steps','examples_seen','validation_every'):
        if baseline['training'][key]!=hybrid['training'][key]:
            raise AssertionError(f'Paired training mismatch: {key}')
    for key in ('n_sequences','steps','seed'):
        if baseline['sampling'][key]!=hybrid['sampling'][key]:
            raise AssertionError(f'Paired sampling mismatch: {key}')
    bundle=load_bundle(data_dir,DEFAULT_CONFIG)
    counts={}
    for name in ('classical','hybrid','classical_matched'):
        model=make_model(name,11)
        counts[name]=trainable_parameters(model)
        if shared_state_sha256(model)!=baseline['initial_shared_state_sha256']:
            raise AssertionError(f'Initial shared state mismatch for {name}')
    if counts!={'classical':1816,'hybrid':1812,'classical_matched':1812}:
        raise AssertionError('Unexpected model parameter counts')
    for record in records.values():
        key=subset_key(record['seed'],record['subset_size'])
        if record['normalization']!=bundle.normalization[key].to_dict():
            raise AssertionError('Recorded normalization differs from the saved subset fit')
        if record['dataset_sha256']!=bundle.metadata['sha256']:
            raise AssertionError('Recorded raw dataset hashes differ')
    reproduction={}
    for name,record in records.items():
        model=make_model(name,record['seed'])
        schedule=NoiseSchedule()
        checkpoint=Path(record['checkpoint'])
        if sha256(checkpoint)!=record['checkpoint_sha256']:
            raise AssertionError(f'Checkpoint checksum mismatch: {name}')
        payload=restore_checkpoint(model,schedule,checkpoint)
        if payload['step']!=record['training']['best_step']:
            raise AssertionError('Restored checkpoint is not the recorded best step')
        regenerated=sample(model,schedule,record['sampling']['n_sequences'],record['sampling']['seed'],
                           record['config']['training']['batch_size'])
        physical=NormalizationStats(**payload['normalization']).inverse_transform(regenerated.samples.numpy())
        saved_path=Path(record['samples_file'])
        if sha256(saved_path)!=record['samples_sha256']:
            raise AssertionError(f'Sample checksum mismatch: {name}')
        saved=np.load(saved_path,allow_pickle=False)
        np.testing.assert_array_equal(physical,saved)
        if saved.shape!=(200,32) or not np.isfinite(saved).all():
            raise AssertionError('Expected 200 finite generated sequences')
        if any(np.any(np.all(bundle.dataset.sequences==row,axis=1)) for row in saved):
            raise AssertionError('A generated row exactly copies a raw dataset row')
        reproduction[name]={'checkpoint_hash':'passed','exact_sample_regeneration':'passed',
                            'no_exact_raw_row_copies':True,'best_checkpoint_step':payload['step']}
    training=hybrid['training']
    for component in ('timestep_encoding','encoder','bottleneck','decoder'):
        if not training['last_model_parameter_change_l2'][component]>0:
            raise AssertionError(f'Hybrid component did not update: {component}')
        for gradients in training['gradient_norms_per_update']:
            if not np.isfinite(gradients[component]) or not gradients[component]>0:
                raise AssertionError(f'Missing/non-finite/zero hybrid gradient: {component}')
    # Four observables, 20 single-use input+weight parameters, two shifts each.
    expected_train=(4*200*len(hybrid['history']) + 4*training['examples_seen']
                    + 2*20*4*training['examples_seen'])
    if training['quantum_submitted_workloads']['observable_expectation_evaluations']!=expected_train:
        raise AssertionError('Training submitted-expectation count mismatch')
    if hybrid['sampling']['quantum_submitted_workloads']['observable_expectation_evaluations']!=20*200*4:
        raise AssertionError('Sampling submitted-expectation count mismatch')
    # Fixed-step budgets are still unspecified. Report coefficients, not an
    # invented training budget or a claim that one epoch is sufficient.
    medians={name:record['training']['median_seconds_per_update'] for name,record in records.items()}
    validations={name:record['training']['validation_seconds']/len(record['history']) for name,record in records.items()}
    costs={'condition_count_per_model':12,'seeds':3,'fractions':4,'approved_full_budget':None,
           'per_model_optimizer_seconds_per_update':medians,
           'mean_seconds_per_validation_pass':validations,
           'per_model_sampling_200_seconds':{name:record['sampling']['seconds'] for name,record in records.items()},
           'hybrid_epoch_optimizer_seconds_estimates':hybrid['optimizer_cost_estimates'],
           'hybrid_one_epoch_each_condition_optimizer_seconds_estimate':
               3*sum(entry['steps_per_epoch'] for entry in hybrid['optimizer_cost_estimates'].values())*medians['hybrid'],
           'hybrid_12_run_sampling_seconds_estimate':12*hybrid['sampling']['seconds'],
           'paired_24_run_optimizer_seconds_per_common_update_budget_unit':12*sum(medians.values()),
           'fixed_update_budget_formula':'For common budget K per condition: 12*K*(classical_median+hybrid_median) optimizer seconds, plus validation, checkpoints, warm-up and sampling.',
           'validation_call_formula':'Each run: 1 + floor(K / validation_every) + int(K % validation_every != 0) full validation passes.',
           'limitations':'Five full-batch updates, one seed/fraction, ideal CPU simulation. Extrapolations exclude checkpoint overhead and have no convergence/hardware-speedup interpretation. Short last batches and system variability may change costs.'}
    files=['models/hybrid_denoiser.py','models/factory.py','models/classical_denoiser.py',
           'training/train.py','training/plotting.py','scripts/check_hybrid_model.py','tests/test_hybrid_denoiser.py']
    return {'status':'passed','created_at_utc':datetime.now(timezone.utc).isoformat(),
            'runs':{name:str(results_dir/f'{run_id}.json') for name,run_id in RUN_IDS.items()},
            'paired_initial_shared_state_sha256':baseline['initial_shared_state_sha256'],
            'paired_data_seeds_schedule_optimizer_and_sampling':'passed','model_parameter_counts':counts,
            'same_parameter_count_is_not_equal_capacity':True,'hybrid_joint_gradient_and_update_checks':'passed',
            'saved_checkpoint_sample_checks':reproduction,
            'cost_estimates':costs,'source_sha256':{name:sha256(ROOT/name) for name in files},
            'full_training_executed':False,'phase6_started':False,'generation_quality_metrics':'Not evaluated'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=ROOT/'results')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data/datasets')
    args=parser.parse_args()
    output=args.results_dir/'phase5_checks.json'
    if output.exists():
        parser.error('Verification report already exists; retain it or choose another results directory')
    try:
        report=verify_pilots(args.results_dir,args.data_dir)
    except Exception as error:
        save_json(output,{'status':'failed','error':f'{type(error).__name__}: {error}'})
        raise
    save_json(output,report)
    print(f'Paired pilot, quantum updates and exact checkpoint/sample reproduction verified: {output}')

if __name__=='__main__': main()
