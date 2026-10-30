"""Execute the predeclared validation-only selection and 36-run Phase 7 grid.

Completed cases are reused only after checksum/configuration checks. Interrupted
training attempts retain their checkpoints and restart deterministically; completed
training can be reused if interruption occurred during sampling or plotting.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'tmp/phase7_matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT/'tmp/phase7_cache'))

import numpy as np
import torch
from config import DEFAULT_CONFIG, DiffusionConfig, config_from_dict
from data.artifacts import load_bundle, sha256, subset_key
from data.normalization import NormalizationStats
from diffusion.noise_schedule import NoiseSchedule
from diffusion.sampling import sample
from evaluation.metrics import evaluate_samples
from experiments.compare import CONFIGURATIONS, verify_pairing, comparison_rows
from models.classical_denoiser import trainable_parameters
from models.factory import make_model, shared_state_sha256, model_state_sha256
from training.plotting import plot_run
from training.train import train_model, restore_checkpoint, quantum_workloads
from utils.reproducibility import seed_everything, environment_metadata, save_json


def write_json(path: Path, payload: dict):
    temporary = path.with_suffix('.pending.json')
    save_json(temporary, payload)
    temporary.replace(path)


def protocol() -> dict:
    config = replace(DEFAULT_CONFIG, diffusion=DiffusionConfig(steps=100,beta_end=.1),
                     model=replace(DEFAULT_CONFIG.model,quantum_backend='batched_statevector'))
    config.validate()
    alpha = float(NoiseSchedule(config.diffusion).alpha_bar[-1])
    return {'version':1,'base_config':config.to_dict(),
            'schedule_selection':'Analytic terminal retained signal variance < 0.01; no quality-based schedule search',
            'terminal_signal_variance':alpha,'torch_threads':1,
            'pilot':{'model':'hybrid','seed':11,'subset_size':1000,'steps':1000},
            'budget_candidates':[500,1000],
            'budget_rule':'Choose 1000 only if best validation noise MSE improves by at least 0.5% after step 500; otherwise 500',
            'lambda_candidates':[0.,.1,1.,10.],
            'lambda_rule':'Among positive weights with validation noise MSE <= 1.10 times the zero-weight control, minimize generated-validation ACF MAE; ties prefer smaller lambda',
            'tuning_seed':11,'tuning_subset_size':1000,'tuning_sampling_seed':2029,
            'validation_every':100,'generated_sequences':200,
            'final_seeds':[11,22,33],'final_subset_sizes':[1000,500,250,100],
            'final_configurations':list(CONFIGURATIONS),'final_run_count':36,
            'checkpoint_selection':'fixed-corruption validation noise MSE',
            'test_access':'Final quality metrics only, after budget and lambda have been frozen',
            'budget_limitation':'A cost-bounded study; completion does not imply convergence',
            'authorization':'User explicitly requested: run phase 7',
            'execution':'Sequential local CPU runs, ideal statevectors, no hardware/cloud access'}


def choose_budget(history: list[dict]) -> dict:
    first = min(row['validation_loss'] for row in history if row['step']<=500)
    final = min(row['validation_loss'] for row in history)
    improvement = (first-final)/first
    return {'selected_steps':1000 if improvement>=.005 else 500,
            'relative_validation_noise_improvement_after_500':improvement,
            'best_validation_noise_mse_by_500':first,'best_validation_noise_mse_by_1000':final,
            'rule':protocol()['budget_rule'],'test_metrics_used':False}


def choose_lambda(candidates: list[dict]) -> dict:
    baseline = next(row['validation_noise_mse'] for row in candidates if row['lambda_acf']==0)
    eligible = [row for row in candidates if row['lambda_acf']>0 and row['validation_noise_mse']<=1.1*baseline
                and row['validation_generated_acf_mae'] is not None
                and np.isfinite(row['validation_generated_acf_mae'])]
    if not eligible:
        raise RuntimeError('No positive guidance weight passed the predeclared validation guard')
    return min(eligible,key=lambda row:(row['validation_generated_acf_mae'],row['lambda_acf']))


class Study:
    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True,exist_ok=True)
        self.manifest_path = self.directory/'status.json'
        expected = json.loads(json.dumps(protocol()))
        path = self.directory/'protocol.json'
        if path.exists():
            if json.loads(path.read_text()) != expected:
                raise ValueError('Frozen protocol differs; use a new study directory')
        else:
            write_json(path,expected)
        self.protocol = expected
        self.config = config_from_dict(expected['base_config'])
        self.bundle = load_bundle(ROOT/'data/datasets',self.config)
        self.manifest = {'status':'running','stage':'initializing','completed_final_runs':0,
                         'total_final_runs':36,'started_at_utc':datetime.now(timezone.utc).isoformat()}
        torch.set_num_threads(1)

    def status(self, **changes):
        self.manifest.update(changes)
        self.manifest['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(self.manifest_path,self.manifest)

    def run_case(self, stage: str, name: str, seed: int, size: int, steps: int,
                 weight: float, run_id: str, reference_split: str | None):
        directory = self.directory/stage
        directory.mkdir(parents=True,exist_ok=True)
        result_path = directory/f'{run_id}.json'
        config = replace(self.config,training=replace(self.config.training,max_steps=steps),
                         loss=replace(self.config.loss,objective='noise_acf' if name=='hybrid_acf' else 'noise',
                                      lambda_acf=weight))
        serialized = json.loads(json.dumps(config.to_dict()))
        if result_path.exists():
            record = json.loads(result_path.read_text())
            if record['config'] != serialized or record['dataset_sha256'] != self.bundle.metadata['sha256']:
                raise ValueError(f'Existing result configuration/provenance mismatch: {run_id}')
            for path_key,hash_key in (('checkpoint','checkpoint_sha256'),('samples_file','samples_sha256')):
                if path_key in record and sha256(Path(record[path_key])) != record[hash_key]:
                    raise ValueError(f'Existing artifact checksum mismatch: {run_id}')
            print(f'Reused completed {run_id}',flush=True)
            return record
        key = subset_key(seed,size)
        stats = self.bundle.normalization[key]
        train = torch.from_numpy(stats.transform(self.bundle.dataset.sequences[self.bundle.subsets[key]])).float()
        validation = torch.from_numpy(stats.transform(self.bundle.dataset.sequences[self.bundle.splits['validation']])).float()
        seed_everything(seed)
        model = make_model(name,seed,config)
        schedule = NoiseSchedule(config.diffusion)
        progress = directory/f'{run_id}_progress.json'
        self.status(current_run=run_id,current_progress_file=str(progress))
        training_path = directory/f'{run_id}_training.json'
        if training_path.exists():
            record = json.loads(training_path.read_text())
            if record['config'] != serialized or sha256(Path(record['checkpoint'])) != record['checkpoint_sha256']:
                raise ValueError(f'Completed training configuration/checksum mismatch: {run_id}')
        else:
            initial_shared,initial_full = shared_state_sha256(model),model_state_sha256(model)
            warm_start = perf_counter()
            model(torch.zeros(16,32),torch.zeros(16,dtype=torch.long)).square().mean().backward()
            model.zero_grad(set_to_none=True)
            warmup_seconds = perf_counter()-warm_start
            checkpoint = directory/'checkpoints'/f'{run_id}_best.pt'
            attempt = 1
            while checkpoint.exists():
                attempt += 1
                checkpoint = directory/'checkpoints'/f'{run_id}_attempt{attempt}_best.pt'
            provenance = {'model_choice':name,'stage':stage,'protocol_sha256':sha256(self.directory/'protocol.json'),
                          'dataset_sha256':self.bundle.metadata['sha256'],
                          'subset_indices':self.bundle.subsets[key].tolist(),
                          'validation_indices':self.bundle.splits['validation'].tolist(),
                          'initial_shared_state_sha256':initial_shared,'initial_model_state_sha256':initial_full}
            trained = train_model(model,schedule,train,validation,config,seed,checkpoint,stats.to_dict(),
                                  self.protocol['validation_every'],provenance,progress_path=progress)
            record = {'purpose':f'Phase 7 {stage}: budgeted experiment, not a convergence claim',
                      'config':serialized,'model_choice':name,'seed':seed,'subset_size':size,
                      'initial_shared_state_sha256':initial_shared,'initial_model_state_sha256':initial_full,
                      'warmup_seconds':warmup_seconds,'training':trained.summary,'history':trained.history,'epochs':trained.epochs,
                      'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),
                      'normalization':stats.to_dict(),'dataset_sha256':self.bundle.metadata['sha256'],
                      'protocol_sha256':sha256(self.directory/'protocol.json'),
                      'environment':environment_metadata(),
                      'execution':{'device':'cpu','dtype':'float32','torch_threads':1,'sequential_runs':True,
                                   'quantum_simulator':'batched ideal statevector with parameter-shift gradients' if name!='classical' else None,
                                   'quantum_hardware_execution':False}}
            write_json(training_path,record)
        restored = restore_checkpoint(model,schedule,Path(record['checkpoint']))
        record['checkpoint_step'] = restored['step']
        if reference_split is not None:
            sampling_seed = self.protocol['tuning_sampling_seed'] if reference_split=='validation' else seed+config.evaluation.sampling_seed_offset
            before = quantum_workloads(model)
            generated = sample(model,schedule,200,sampling_seed,config.training.batch_size)
            physical = stats.inverse_transform(generated.samples.numpy())
            samples_path = directory/'samples'/f'{run_id}.npy'
            samples_path.parent.mkdir(parents=True,exist_ok=True)
            if samples_path.exists():
                if not np.array_equal(np.load(samples_path,allow_pickle=False),physical):
                    raise ValueError(f'Partial saved samples differ: {run_id}')
            else:
                np.save(samples_path,physical,allow_pickle=False)
            record.update(samples_file=str(samples_path),samples_sha256=sha256(samples_path),
                          reference_split=reference_split,
                          generation_quality_metrics=evaluate_samples(physical,self.bundle.dataset.sequences[self.bundle.splits[reference_split]],10),
                          sampling={'seconds':generated.seconds,'n_sequences':200,'steps':schedule.steps,'seed':sampling_seed,
                                    'step_max_abs_normalized':generated.step_max_abs,
                                    'quantum_submitted_workloads':{k:v-before[k] for k,v in quantum_workloads(model).items()}})
            figures = plot_run(record['history'],physical,self.bundle.dataset.time,directory/'figures',run_id,
                               model_label={'classical':'Classical','hybrid':'Hybrid','hybrid_acf':'Hybrid + ACF'}[name],smoke=False)
            record['figures_sha256'] = {str(path):sha256(path) for path in figures}
        write_json(result_path,record)
        print(f'Completed {run_id}: {steps} updates, best validation MSE={record["training"]["best_validation_loss"]:.6f}',flush=True)
        return record

    def run(self):
        self.status(stage='validation_budget_pilot')
        pilot = self.run_case('pilot','hybrid',11,1000,1000,0.,'budget_pilot',None)
        budget_record = choose_budget(pilot['history'])
        budget = budget_record['selected_steps']
        write_json(self.directory/'budget_selection.json',budget_record)
        self.status(stage='validation_lambda_tuning',selected_steps=budget)
        tuning = []
        for weight in self.protocol['lambda_candidates']:
            label = str(weight).replace('.','p')
            record = self.run_case('tuning','hybrid_acf',11,1000,budget,weight,f'lambda_{label}','validation')
            tuning.append({'lambda_acf':weight,'validation_noise_mse':record['training']['best_validation_loss'],
                           'validation_generated_acf_mae':record['generation_quality_metrics']['acf_mae'],
                           'validation_generated_wasserstein_1':record['generation_quality_metrics']['marginal_wasserstein_1']})
        selected = choose_lambda(tuning)
        weight = selected['lambda_acf']
        selection = {'selected_lambda_acf':weight,'selected_steps':budget,'candidates':tuning,
                     'selection_rule':self.protocol['lambda_rule'],'test_metrics_used':False,
                     'frozen_at_utc':datetime.now(timezone.utc).isoformat()}
        frozen = self.directory/'frozen_selection.json'
        if frozen.exists():
            previous = json.loads(frozen.read_text())
            if any(previous[key]!=selection[key] for key in ('selected_lambda_acf','selected_steps','candidates')):
                raise ValueError('Selection differs from previously frozen study')
        else:
            write_json(frozen,selection)
        self.status(stage='final_experiments',selected_lambda_acf=weight,selected_steps=budget)
        records,verification = [],{}
        for seed in self.protocol['final_seeds']:
            for size in self.protocol['final_subset_sizes']:
                prefix = f'phase7_seed{seed}_n{size}'
                group = []
                for name in CONFIGURATIONS:
                    record = self.run_case('final',name,seed,size,budget,weight,prefix+'_'+name,'test')
                    group.append(record); records.append(record)
                    self.status(completed_final_runs=len(records))
                verify_pairing(group)
                write_json(self.directory/'final'/f'{prefix}_comparison.json',
                           {'pairing':'passed','pilot':False,'rows':comparison_rows(group)})
        self.status(stage='reproduction_verification')
        from scripts.check_three_configurations import verify_runs
        for seed in self.protocol['final_seeds']:
            for size in self.protocol['final_subset_sizes']:
                prefix = f'phase7_seed{seed}_n{size}'
                verification[prefix] = verify_runs(self.directory/'final',prefix,ROOT/'data/datasets')
                print(f'Verified exact checkpoint/sample reproduction: {prefix}',flush=True)
        write_json(self.directory/'verification.json',{'status':'passed','groups':verification})
        import csv
        rows = comparison_rows(records)
        with (self.directory/'final_runs.csv').open('w',newline='') as handle:
            writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        write_json(self.directory/'completion.json',
                   {'status':'complete','final_runs':len(records),'verified_groups':len(verification),
                    'selected_steps':budget,'selected_lambda_acf':weight,
                    'schedule':self.protocol['base_config']['diffusion'],
                    'terminal_signal_variance':self.protocol['terminal_signal_variance'],
                    'test_used_for_selection':False,'protocol_sha256':sha256(self.directory/'protocol.json'),
                    'frozen_selection_sha256':sha256(frozen),
                    'effective_epochs_by_subset':{str(size):[r['training']['effective_epochs'] for r in records if r['subset_size']==size]
                                                  for size in self.protocol['final_subset_sizes']},
                    'convergence_claim':False,'next_phase':'Phase 8: aggregation, interpretation and report'})
        self.status(status='complete',stage='complete',completed_final_runs=36)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=ROOT/'results/phase7')
    parser.add_argument('--prepare-only',action='store_true')
    args = parser.parse_args()
    study = Study(args.results_dir)
    if args.prepare_only:
        print(f'Frozen protocol prepared: {study.directory / "protocol.json"}')
        return
    try:
        study.run()
    except BaseException:
        study.status(status='failed',error=traceback.format_exc())
        raise


if __name__ == '__main__': main()
