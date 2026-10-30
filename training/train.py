"""Shared CPU DDPM trainer; run explicitly with a smoke or approved budget."""
import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
import sys
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from config import DEFAULT_CONFIG, ExperimentConfig, LossConfig
from data.artifacts import load_bundle, subset_key, sha256
from models.classical_denoiser import trainable_parameters
from models.factory import MODEL_TYPES, make_model, shared_state_sha256, model_state_sha256
from diffusion.noise_schedule import NoiseSchedule
from diffusion.sampling import sample
from training.plotting import plot_run
from training.losses import diffusion_losses
from evaluation.metrics import evaluate_samples
from utils.reproducibility import seed_everything, environment_metadata, save_json


def stream_seeds(run_seed: int, validation_seed: int) -> dict[str,int]:
    """Separate training RNG streams; fixed validation stream across models."""
    def derive(seed, stream):
        return int(np.random.SeedSequence([seed,stream]).generate_state(1)[0])
    return {"shuffle":derive(run_seed,1),"timesteps":derive(run_seed,2),"noise":derive(run_seed,3),
            "validation_timesteps":derive(validation_seed,1),"validation_noise":derive(validation_seed,2)}


def generator(seed: int) -> torch.Generator:
    return torch.Generator(device='cpu').manual_seed(seed)


def validate_data(values: torch.Tensor) -> None:
    if values.ndim != 2 or values.shape[1] != 32 or len(values)==0:
        raise ValueError('Training/validation data must have nonempty (N,32) shape')
    if values.device.type != 'cpu' or values.dtype != torch.float32 or not torch.isfinite(values).all():
        raise ValueError('Initial training data must be finite CPU float32')


def validation_draws(values: torch.Tensor, steps: int, seeds: dict[str,int]) -> tuple[torch.Tensor,torch.Tensor]:
    return (torch.randint(steps,(len(values),),generator=generator(seeds['validation_timesteps'])),
            torch.randn(values.shape,generator=generator(seeds['validation_noise'])))


@torch.no_grad()
def validation_components(model: nn.Module, schedule: NoiseSchedule, values: torch.Tensor,
                          timestep: torch.Tensor, noise: torch.Tensor, batch_size: int,
                          loss_config: LossConfig = LossConfig()) -> dict[str, float]:
    """Fixed corruption; report all losses but select all arms by noise MSE."""
    was_training = model.training
    model.eval()
    totals = dict.fromkeys(("noise_loss", "acf_loss", "total_loss"), 0.)
    try:
        for begin in range(0, len(values), batch_size):
            clean = values[begin:begin + batch_size]
            time = timestep[begin:begin + batch_size]
            target = noise[begin:begin + batch_size]
            noisy = schedule.add_noise(clean, time, target)
            prediction = model(noisy, time)
            if prediction.shape != target.shape or not torch.isfinite(prediction).all():
                raise ValueError('Denoiser must predict finite noise with shape (B,32)')
            losses = diffusion_losses(prediction, target, noisy, clean, time, schedule, loss_config)
            for key, value in losses.items():
                totals[key] += float(value) * len(clean)
    finally:
        model.train(was_training)
    result = {key: value / len(values) for key, value in totals.items()}
    if not all(np.isfinite(value) for value in result.values()):
        raise FloatingPointError('Non-finite validation loss')
    return result


def validation_loss(model: nn.Module, schedule: NoiseSchedule, values: torch.Tensor,
                    timestep: torch.Tensor, noise: torch.Tensor, batch_size: int) -> float:
    return validation_components(model, schedule, values, timestep, noise, batch_size)['noise_loss']


def quantum_workloads(model: nn.Module) -> dict[str, int]:
    estimator = getattr(getattr(model, 'bottleneck', None), 'estimator', None)
    return dict(estimator.counts) if estimator is not None else {}


def component_gradient_norms(model: nn.Module) -> dict[str, float]:
    squared = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            component = name.split('.')[0]
            squared[component] = squared.get(component, 0.) + float(parameter.grad.detach().double().square().sum())
    return {name: value**.5 for name, value in squared.items()}


def component_parameter_changes(model: nn.Module, initial: dict[str, torch.Tensor]) -> dict[str, float]:
    squared = {}
    for name, parameter in model.named_parameters():
        component = name.split('.')[0]
        delta = parameter.detach() - initial[name]
        squared[component] = squared.get(component, 0.) + float(delta.double().square().sum())
    return {name: value**.5 for name, value in squared.items()}


@dataclass
class TrainingResult:
    history: list[dict]
    epochs: list[dict]
    summary: dict
    checkpoint: Path


def train_model(model: nn.Module, schedule: NoiseSchedule, train: torch.Tensor, validation: torch.Tensor,
                config: ExperimentConfig, run_seed: int, checkpoint: Path, normalization: dict,
                validation_every: int = 63, provenance: dict | None = None,
                progress_path: Path | None = None) -> TrainingResult:
    """One pipeline for any compatible denoiser. Exactly one explicit budget.

    Records partial epochs, retains final short batches and uses separate local
    generators for shuffle/timestep/noise. Best checkpoint can be step 0 if the
    initial model beats every smoke checkpoint; this is reported explicitly.
    """
    config.validate()
    validate_data(train); validate_data(validation)
    if schedule.steps!=config.diffusion.steps:
        raise ValueError('Schedule step count differs from experiment configuration')
    setting=config.training
    if (setting.epochs is None)==(setting.max_steps is None):
        raise ValueError('Specify exactly one positive epochs or max_steps budget')
    if validation_every<=0:
        raise ValueError('Validation interval must be positive')
    if any(p.device.type!='cpu' or p.dtype!=torch.float32 for p in model.parameters()):
        raise ValueError('Initial denoiser parameters must be CPU float32')
    seeds=stream_seeds(run_seed,config.evaluation.validation_seed)
    shuffle_rng=generator(seeds['shuffle']); time_rng=generator(seeds['timesteps']); noise_rng=generator(seeds['noise'])
    fixed_time,fixed_noise=validation_draws(validation,schedule.steps,seeds)
    optimizer=torch.optim.Adam(model.parameters(),lr=setting.learning_rate)
    checkpoint=Path(checkpoint)
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    if checkpoint.exists():
        raise FileExistsError('Checkpoint already exists; use a new run ID/directory')
    epoch_limit=setting.epochs
    budget=setting.max_steps if setting.max_steps is not None else epoch_limit*((len(train)+setting.batch_size-1)//setting.batch_size)
    initial_parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
    initial_quantum_counts = quantum_workloads(model)
    gradient_history = []
    loss_history = []
    history=[]; epoch_records=[]; update_seconds=[]; step=0; examples_seen=0
    checkpoint_seconds=0.; validation_seconds=0.
    start=perf_counter()
    val_start=perf_counter()
    initial_components=validation_components(model,schedule,validation,fixed_time,fixed_noise,setting.batch_size,config.loss)
    initial_val=initial_components["noise_loss"]
    validation_seconds+=perf_counter()-val_start
    best_loss=initial_val; best_step=0
    history.append({'step':0,'train_loss':None,'validation_loss':initial_val,'examples_seen':0,
                    **{'validation_'+key:value for key,value in initial_components.items()}})

    def save_checkpoint(epoch: int, cursor: int):
        nonlocal checkpoint_seconds
        save_start=perf_counter()
        payload={'schema_version':1,'model_type':type(model).__name__,'model_state':model.state_dict(),
                 'optimizer_state':optimizer.state_dict(),'schedule_state':schedule.state_dict(),
                 'step':step,'epoch':epoch,'batch_cursor':cursor,'best_validation_loss':best_loss,
                 'config':config.to_dict(),'checkpoint_selection':'validation_noise_mse','normalization':normalization,'run_seed':run_seed,'validation_every':validation_every,
                 'stream_seeds':seeds,'rng_states':{'shuffle':shuffle_rng.get_state(),
                       'timesteps':time_rng.get_state(),'noise':noise_rng.get_state(), 'torch_global':torch.get_rng_state()},
                 'provenance':provenance or {}}
        temporary=checkpoint.with_suffix('.tmp')
        torch.save(payload,temporary)
        temporary.replace(checkpoint)
        checkpoint_seconds+=perf_counter()-save_start

    save_checkpoint(0,0)
    interval_error=0.; interval_elements=0
    interval_components = dict.fromkeys(("noise_loss", "acf_loss", "total_loss"), 0.)
    interval_examples = 0
    epoch=0
    while step<budget:
        epoch+=1
        epoch_start=perf_counter()
        order=torch.randperm(len(train),generator=shuffle_rng)
        epoch_error=0.; epoch_elements=0; epoch_examples=0; epoch_steps=0
        for begin in range(0,len(train),setting.batch_size):
            if step>=budget: break
            update_start=perf_counter()
            clean=train[order[begin:begin+setting.batch_size]]
            timestep=torch.randint(schedule.steps,(len(clean),),generator=time_rng)
            noise=torch.randn(clean.shape,generator=noise_rng)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            noisy=schedule.add_noise(clean,timestep,noise)
            prediction=model(noisy,timestep)
            if prediction.shape!=noise.shape or not torch.isfinite(prediction).all():
                raise ValueError('Denoiser must predict finite noise with shape (B,32)')
            losses=diffusion_losses(prediction,noise,noisy,clean,timestep,schedule,config.loss)
            loss=losses["total_loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss_history.append({"step":step+1, **{key:float(value.detach()) for key,value in losses.items()}})
            for key,value in losses.items():
                interval_components[key] += float(value.detach()) * len(clean)
            interval_examples += len(clean)
            loss.backward()
            gradients=[p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or any(not torch.isfinite(g).all() for g in gradients):
                raise FloatingPointError('Missing or non-finite model gradients')
            gradient_history.append(component_gradient_norms(model))
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError('Non-finite optimized parameters')
            update_seconds.append(perf_counter()-update_start)
            step+=1; epoch_steps+=1; examples_seen+=len(clean); epoch_examples+=len(clean)
            squared_error=float(losses["noise_loss"].detach())*clean.numel()
            epoch_error+=squared_error; epoch_elements+=clean.numel()
            interval_error+=squared_error; interval_elements+=clean.numel()
            if step%validation_every==0 or step==budget:
                val_start=perf_counter()
                components=validation_components(model,schedule,validation,fixed_time,fixed_noise,setting.batch_size,config.loss)
                val=components["noise_loss"]
                validation_seconds+=perf_counter()-val_start
                history.append({'step':step,'train_loss':interval_error/interval_elements,
                                'validation_loss':val,'examples_seen':examples_seen,
                                **{'train_'+key:value/interval_examples for key,value in interval_components.items()},
                                **{'validation_'+key:value for key,value in components.items()}})
                interval_error=0.; interval_elements=0
                interval_components=dict.fromkeys(interval_components,0.)
                interval_examples=0
                if val<best_loss:
                    best_loss=val; best_step=step
                    save_checkpoint(epoch,begin+len(clean))
                if progress_path is not None:
                    save_json(progress_path, {'step':step,'budget':budget,'best_step':best_step,
                                              'best_validation_noise_mse':best_loss,
                                              'validation':components,'elapsed_seconds':perf_counter()-start})
        epoch_records.append({'epoch':epoch,'complete':epoch_examples==len(train),'steps':epoch_steps,
                              'examples':epoch_examples,'train_loss':epoch_error/epoch_elements,
                              'seconds':perf_counter()-epoch_start})
    seconds=perf_counter()-start
    summary={'objective':config.loss.objective,'lambda_acf':config.loss.lambda_acf,
             'effective_lambda_acf':config.loss.lambda_acf if config.loss.objective=='noise_acf' else 0.,
             'checkpoint_selection':'validation_noise_mse','losses_per_update':loss_history,'steps':step,'examples_seen':examples_seen,'validation_every':validation_every,'effective_epochs':examples_seen/len(train),
             'completed_epochs':sum(row['complete'] for row in epoch_records),
             'trainable_parameters':trainable_parameters(model),'training_seconds':seconds,
             'update_seconds':update_seconds,'optimizer_loop_seconds':sum(update_seconds),
             'mean_seconds_per_update':float(np.mean(update_seconds)),
             'median_seconds_per_update':float(np.median(update_seconds)),
             'validation_seconds':validation_seconds,'checkpoint_seconds':checkpoint_seconds,
             'initial_validation_loss':initial_val,'best_validation_loss':best_loss,'best_step':best_step,
             'last_validation_loss':history[-1]['validation_loss'],'stream_seeds':seeds,
             'gradient_norms_per_update':gradient_history,
             'last_model_parameter_change_l2':component_parameter_changes(model, initial_parameters),
             'quantum_submitted_workloads':{key: value-initial_quantum_counts[key]
                                            for key,value in quantum_workloads(model).items()}}
    return TrainingResult(history,epoch_records,summary,checkpoint)


def restore_checkpoint(model: nn.Module, schedule: NoiseSchedule, checkpoint: Path) -> dict:
    """Load tensor/basic-type checkpoint safely; no arbitrary pickle objects."""
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if payload.get('schema_version')!=1 or payload['model_type']!=type(model).__name__:
        raise ValueError('Checkpoint schema/model type mismatch')
    model.load_state_dict(payload['model_state'])
    schedule.load_state_dict(payload['schedule_state'])
    return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    budget=parser.add_mutually_exclusive_group(required=True)
    budget.add_argument('--smoke',action='store_true',help='Only the five-step configured pilot')
    budget.add_argument('--max-steps',type=int,help='Explicit approved update budget')
    budget.add_argument('--epochs',type=int,help='Explicit approved epoch budget')
    parser.add_argument('--model',choices=MODEL_TYPES,default='classical')
    parser.add_argument('--lambda-acf',type=float,default=None,help='Guided hybrid weight; default 1.0 is provisional')
    parser.add_argument('--seed',type=int,choices=DEFAULT_CONFIG.training.seeds,default=11)
    parser.add_argument('--subset-size',type=int,choices=DEFAULT_CONFIG.dataset.subset_sizes,default=1000)
    parser.add_argument('--validation-every',type=int,default=None)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data/datasets')
    parser.add_argument('--results-dir',type=Path,default=ROOT/'results')
    parser.add_argument('--run-id',default=None)
    args=parser.parse_args()
    args.run_id=args.run_id or f"{args.model}_{'smoke' if args.smoke else 'train'}_seed{args.seed}_n{args.subset_size}"
    if not args.run_id or any(character not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for character in args.run_id):
        parser.error('Run ID must contain only letters, digits, underscore or hyphen')
    steps=DEFAULT_CONFIG.training.pilot_steps if args.smoke else args.max_steps
    if args.lambda_acf is not None and args.model != 'hybrid_acf':
        parser.error('--lambda-acf is only valid with --model hybrid_acf')
    loss_config=replace(DEFAULT_CONFIG.loss,objective='noise_acf' if args.model=='hybrid_acf' else 'noise',
                        lambda_acf=DEFAULT_CONFIG.loss.lambda_acf if args.lambda_acf is None else args.lambda_acf)
    config=replace(DEFAULT_CONFIG,training=replace(DEFAULT_CONFIG.training,max_steps=steps,epochs=args.epochs),loss=loss_config)
    config.validate()
    results=args.results_dir
    summary_path=results/f'{args.run_id}.json'
    checkpoint=results/'checkpoints'/f'{args.run_id}_best.pt'
    if summary_path.exists() or checkpoint.exists():
        parser.error('Run artifacts already exist; choose a new run ID')
    bundle=load_bundle(args.data_dir,config)
    normalized=bundle.normalized(args.seed,args.subset_size)
    train=torch.from_numpy(normalized['train']).float()
    validation=torch.from_numpy(normalized['validation']).float()
    seed_everything(args.seed)
    model=make_model(args.model,args.seed,config)
    initial_shared_hash=shared_state_sha256(model)
    initial_model_hash=model_state_sha256(model)
    initial_parameters={name: p.detach().clone() for name,p in model.named_parameters()}
    schedule=NoiseSchedule(config.diffusion)
    warmup_start=perf_counter()
    model(torch.zeros(16,32),torch.zeros(16,dtype=torch.long)).square().mean().backward()
    model.zero_grad(set_to_none=True)
    warmup_seconds=perf_counter()-warmup_start
    interval=args.validation_every if args.validation_every is not None else (1 if args.smoke else 63)
    provenance={'model_choice':args.model,'initial_shared_state_sha256':initial_shared_hash,'initial_model_state_sha256':initial_model_hash,
                'dataset_sha256':bundle.metadata['sha256'],'subset_indices':bundle.subsets[subset_key(args.seed,args.subset_size)].tolist(),
                'validation_indices':bundle.splits['validation'].tolist(),'environment':environment_metadata(),
                'purpose':'smoke test, not convergence or quality evidence' if args.smoke else 'explicitly budgeted training'}
    before_training_counts=quantum_workloads(model)
    trained=train_model(model,schedule,train,validation,config,args.seed,checkpoint,
                        bundle.normalization[subset_key(args.seed,args.subset_size)].to_dict(),interval,provenance)
    restored=restore_checkpoint(model,schedule,checkpoint)
    best_parameter_changes=component_parameter_changes(model,initial_parameters)
    before_sampling_counts=quantum_workloads(model)
    generated=sample(model,schedule,config.evaluation.generated_sequences,
                     args.seed+config.evaluation.sampling_seed_offset,config.training.batch_size)
    physical=bundle.normalization[subset_key(args.seed,args.subset_size)].inverse_transform(generated.samples.numpy())
    samples_path=results/'samples'/f'{args.run_id}.npy'
    samples_path.parent.mkdir(parents=True,exist_ok=True)
    np.save(samples_path,physical,allow_pickle=False)
    figures=plot_run(trained.history,physical,bundle.dataset.time,results/'figures',args.run_id,
                     model_label={'classical':'Classical','hybrid':'Hybrid','hybrid_acf':'Hybrid + ACF','classical_matched':'Parameter-matched classical'}[args.model],
                     smoke=args.smoke)
    median=trained.summary['median_seconds_per_update']
    # Update-only estimates exclude validation/checkpoints; assumptions stay explicit.
    batch_size=config.training.batch_size
    estimates={str(size):{'steps_per_epoch':(size+batch_size-1)//batch_size,
                         'optimizer_seconds_per_epoch_estimate':((size+batch_size-1)//batch_size)*median}
               for size in config.dataset.subset_sizes}
    record={'purpose':provenance['purpose'],'config':config.to_dict(),'model_choice':args.model,
            'initial_shared_state_sha256':initial_shared_hash,'initial_model_state_sha256':initial_model_hash,'best_model_parameter_change_l2':best_parameter_changes,
            'warmup_quantum_submitted_workloads':before_training_counts,
            'execution':{'device':'cpu','dtype':'float32','torch_threads':torch.get_num_threads(),
                         'torch_interop_threads':torch.get_num_interop_threads(),
                         'quantum_simulator':'ideal StatevectorEstimator, zero precision' if args.model in ('hybrid','hybrid_acf') else None,
                         'quantum_hardware_execution':False},'seed':args.seed,'subset_size':args.subset_size,
            'warmup_seconds':warmup_seconds,'training':trained.summary,'history':trained.history,'epochs':trained.epochs,
            'sampling':{'seconds':generated.seconds,'n_sequences':len(physical),'steps':schedule.steps,
                        'seed':generated.seed,'step_max_abs_normalized':generated.step_max_abs,
                        'quantum_submitted_workloads':{key:value-before_sampling_counts[key] for key,value in quantum_workloads(model).items()},
                        'max_abs_physical':float(np.abs(physical).max())},
            'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),'checkpoint_step':restored['step'],
            'normalization':restored['normalization'],'environment':environment_metadata(),
            'dataset_sha256':bundle.metadata['sha256'],'samples_file':str(samples_path),'samples_sha256':sha256(samples_path),
            'figures_sha256':{path.name:sha256(path) for path in figures},
            'optimizer_cost_estimates':estimates,
            'estimate_limitations':'Median measured updates; optimizer-only extrapolation from this run, excluding validation/checkpoints. No convergence or hardware speedup claim.',
            'generation_quality_metrics':evaluate_samples(physical,bundle.dataset.sequences[bundle.splits['test']],config.evaluation.max_acf_lag),
            'quality_interpretation':'Pilot diagnostics only; not convergence or superiority evidence' if args.smoke else 'Held-out evaluation; assess across paired repeated runs'}
    save_json(summary_path,record)
    print(f"Completed {trained.summary['steps']} updates; parameters={trainable_parameters(model)}; best checkpoint step={restored['step']}")
    print(f"Training={trained.summary['training_seconds']:.6f}s; sampling {len(physical)} sequences={generated.seconds:.6f}s")
    print(f'Run evidence: {summary_path}')

if __name__=='__main__':
    main()
