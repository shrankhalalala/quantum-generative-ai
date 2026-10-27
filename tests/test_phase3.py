"""Classical model, DDPM equations, sampling and shared trainer correctness."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from torch import nn
from config import DEFAULT_CONFIG
from models.classical_denoiser import ClassicalDenoiser, Denoiser, trainable_parameters
from diffusion.noise_schedule import NoiseSchedule
from diffusion.sampling import sample
from training.train import train_model, restore_checkpoint, stream_seeds, validation_draws, validation_loss
from utils.reproducibility import seed_everything


class ZeroDenoiser(nn.Module):
    def forward(self, values, timestep):
        return torch.zeros_like(values)


class ModelTests(unittest.TestCase):
    def setUp(self):
        seed_everything(11)
        self.model=ClassicalDenoiser()

    def test_shape_capacity_and_timestep_dependence(self):
        self.assertEqual(trainable_parameters(self.model),1816)
        values=torch.ones(2,32)
        output=self.model(values,torch.tensor([0,19]))
        self.assertEqual(output.shape,(2,32))
        self.assertTrue(torch.isfinite(output).all())
        self.assertFalse(torch.equal(output[0],output[1]))
        self.assertEqual(trainable_parameters(Denoiser(nn.Identity())),1796)

    def test_gradients_reach_every_component_and_input(self):
        values=torch.randn(3,32,requires_grad=True)
        self.model(values,torch.tensor([0,5,19])).square().mean().backward()
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertGreater(float(values.grad.abs().sum()),0.)
        for name,parameter in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()),0.)

    def test_invalid_model_inputs(self):
        for values,time in ((torch.zeros(2,31),torch.tensor([0,0])),
                            (torch.zeros(2,32),torch.tensor([0.,0.])),
                            (torch.zeros(2,32),torch.tensor([0,20])),
                            (torch.zeros(2,32),torch.tensor([0])),
                            (torch.full((2,32),float('nan')),torch.tensor([0,0]))):
            with self.assertRaises(ValueError): self.model(values,time)


class ScheduleTests(unittest.TestCase):
    def setUp(self): self.schedule=NoiseSchedule()

    def test_buffers_match_analytic_schedule(self):
        beta=np.linspace(.0001,.02,20)
        alpha=1-beta
        cumulative=np.cumprod(alpha)
        previous=np.r_[1.,cumulative[:-1]]
        np.testing.assert_allclose(self.schedule.beta.numpy(),beta,rtol=1e-7)
        np.testing.assert_allclose(self.schedule.alpha_bar.numpy(),cumulative,rtol=1e-7)
        np.testing.assert_allclose(self.schedule.posterior_variance.numpy(),beta*(1-previous)/(1-cumulative),rtol=1e-7)
        self.assertEqual(float(self.schedule.posterior_variance[0]),0.)
        self.assertEqual(len(list(self.schedule.parameters())),0)
        self.assertGreater(float(self.schedule.alpha_bar[-1]),.8)

    def test_forward_noising_known_formula(self):
        clean=torch.arange(96,dtype=torch.float32).reshape(3,32)/100
        noise=torch.full_like(clean,.3)
        time=torch.tensor([0,7,19])
        cumulative=self.schedule.alpha_bar[time].unsqueeze(1)
        expected=cumulative.sqrt()*clean+(1-cumulative).sqrt()*noise
        torch.testing.assert_close(self.schedule.add_noise(clean,time,noise),expected,rtol=0,atol=0)

    def test_reverse_mean_and_variance_formula(self):
        noisy=torch.full((2,32),.4)
        prediction=torch.full_like(noisy,.2)
        noise=torch.full_like(noisy,.7)
        time=torch.tensor([0,19])
        a=self.schedule.alpha[time,None]; b=self.schedule.beta[time,None]; abar=self.schedule.alpha_bar[time,None]
        expected=(noisy-b*prediction/(1-abar).sqrt())/a.sqrt()
        torch.testing.assert_close(self.schedule.reverse_mean(noisy,time,prediction),expected,rtol=0,atol=0)
        expected+=self.schedule.posterior_variance[time,None].sqrt()*noise
        torch.testing.assert_close(self.schedule.reverse_step(noisy,time,prediction,noise),expected,rtol=0,atol=0)

    def test_last_step_noise_has_no_effect(self):
        values=torch.ones(2,32); time=torch.zeros(2,dtype=torch.long)
        a=self.schedule.reverse_step(values,time,torch.zeros_like(values),torch.ones_like(values))
        b=self.schedule.reverse_step(values,time,torch.zeros_like(values),torch.full_like(values,10.))
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_invalid_schedule_tensors(self):
        values=torch.zeros(2,32)
        for time,noise in ((torch.tensor([-1,0]),values),(torch.tensor([0.,1.]),values),
                           (torch.tensor([0,0]),torch.zeros(2,31)),
                           (torch.tensor([0,0]),torch.full_like(values,float('nan')))):
            with self.assertRaises(ValueError): self.schedule.add_noise(values,time,noise)


class SamplingTests(unittest.TestCase):
    def test_seeded_shape_repeatability_and_mode_restoration(self):
        seed_everything(11)
        model=ClassicalDenoiser(); model.train()
        schedule=NoiseSchedule()
        a=sample(model,schedule,7,123,batch_size=3)
        b=sample(model,schedule,7,123,batch_size=3)
        c=sample(model,schedule,7,124,batch_size=3)
        self.assertTrue(model.training)
        self.assertEqual(a.samples.shape,(7,32))
        self.assertEqual(len(a.step_max_abs),20)
        self.assertFalse(a.samples.requires_grad)
        self.assertTrue(torch.equal(a.samples,b.samples))
        self.assertFalse(torch.equal(a.samples,c.samples))
        self.assertTrue(torch.isfinite(a.samples).all())

    def test_explicit_gaussian_draws_match_independent_reverse_calculation(self):
        schedule=NoiseSchedule(); model=ZeroDenoiser()
        rng=torch.Generator().manual_seed(5)
        initial=torch.randn(3,32,generator=rng)
        noise=torch.randn(19,3,32,generator=rng)
        initial_copy=initial.clone()
        result=sample(model,schedule,3,123,batch_size=2,initial_noise=initial,reverse_noise=noise)
        expected=initial.clone()
        for time in range(19,-1,-1):
            expected=expected/schedule.alpha[time].sqrt()
            if time>0: expected+=schedule.posterior_variance[time].sqrt()*noise[time-1]
        torch.testing.assert_close(result.samples,expected,rtol=0,atol=0)
        self.assertTrue(torch.equal(initial,initial_copy))
        paired=sample(model,schedule,3,456,batch_size=3,initial_noise=initial,reverse_noise=noise)
        torch.testing.assert_close(result.samples,paired.samples,rtol=0,atol=0)

    def test_bad_outputs_and_instability_restore_mode(self):
        class BadDenoiser(nn.Module):
            def forward(self,x,t): return torch.full_like(x,float('nan'))
        model=BadDenoiser(); model.train()
        with self.assertRaises(ValueError): sample(model,NoiseSchedule(),2,11)
        self.assertTrue(model.training)
        zero=ZeroDenoiser()
        with self.assertRaisesRegex(FloatingPointError,'Unstable'):
            sample(zero,NoiseSchedule(),2,11,max_abs=.001)
        with self.assertRaises(ValueError): sample(zero,NoiseSchedule(),2,11,initial_noise=torch.ones(2,31))


class TrainingTests(unittest.TestCase):
    def fixtures(self, steps=3, epochs=None, seed=11):
        seed_everything(seed)
        model=ClassicalDenoiser(); schedule=NoiseSchedule()
        rng=torch.Generator().manual_seed(4)
        train=torch.randn(17,32,generator=rng)
        validation=torch.randn(9,32,generator=rng)
        config=replace(DEFAULT_CONFIG,training=replace(DEFAULT_CONFIG.training,max_steps=steps,epochs=epochs))
        return model,schedule,train,validation,config

    def train(self, path, steps=3, epochs=None, validation_every=1):
        model,schedule,train,val,config=self.fixtures(steps,epochs)
        result=train_model(model,schedule,train,val,config,11,path,
                           {'mean':0.,'std':1.,'n_sequences':17,'sequence_length':32,'ddof':0},validation_every)
        return model,schedule,result

    def test_validation_fixed_draws_weighting_and_rng_independence(self):
        model,schedule,train,val,config=self.fixtures()
        seeds=stream_seeds(11,2028)
        time,noise=validation_draws(val,20,seeds)
        state=torch.get_rng_state().clone()
        a=validation_loss(model,schedule,val,time,noise,4)
        b=validation_loss(model,schedule,val,time,noise,9)
        self.assertAlmostEqual(a,b,places=6)
        expected=float((model(schedule.add_noise(val,time,noise),time)-noise).square().mean().detach())
        self.assertAlmostEqual(a,expected,places=6)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        model.train()
        validation_loss(model,schedule,val,time,noise,4)
        self.assertTrue(model.training)
        other=stream_seeds(22,2028)
        repeat_time,repeat_noise=validation_draws(val,20,other)
        self.assertTrue(torch.equal(time,repeat_time))
        self.assertTrue(torch.equal(noise,repeat_noise))

    def test_epoch_budget_retains_short_last_batch_and_updates_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            model,schedule,train,val,config=self.fixtures(None,1)
            before={name:p.detach().clone() for name,p in model.named_parameters()}
            result=train_model(model,schedule,train,val,config,11,Path(directory)/'best.pt',
                               {'mean':0.,'std':1.,'n_sequences':17,'sequence_length':32,'ddof':0},1)
            self.assertEqual(result.summary['steps'],2)
            self.assertEqual(result.summary['examples_seen'],17)
            self.assertEqual(result.summary['completed_epochs'],1)
            self.assertTrue(result.epochs[0]['complete'])
            self.assertTrue(any(not torch.equal(p,before[name]) for name,p in model.named_parameters()))

    def test_repeated_training_and_validation_frequency_do_not_change_final_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            a,_,result_a=self.train(Path(directory)/'a.pt')
            b,_,result_b=self.train(Path(directory)/'b.pt')
            c,_,_=self.train(Path(directory)/'c.pt',validation_every=2)
            self.assertEqual(result_a.history,result_b.history)
            for name,value in a.state_dict().items():
                self.assertTrue(torch.equal(value,b.state_dict()[name]))
                self.assertTrue(torch.equal(value,c.state_dict()[name]))
            self.assertEqual(result_a.summary['examples_seen'],33) # 16+1+16, no dropped last batch
            self.assertFalse(result_a.epochs[-1]['complete'])

    def test_checkpoint_is_best_and_reloads_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            model,schedule,result=self.train(Path(directory)/'best.pt')
            payload=restore_checkpoint(model,schedule,result.checkpoint)
            self.assertEqual(payload['step'],result.summary['best_step'])
            self.assertEqual(payload['best_validation_loss'],min(row['validation_loss'] for row in result.history))
            clone=ClassicalDenoiser(); cloned_schedule=NoiseSchedule()
            restore_checkpoint(clone,cloned_schedule,result.checkpoint)
            x=torch.ones(2,32); time=torch.tensor([0,19])
            self.assertTrue(torch.equal(model(x,time),clone(x,time)))
            self.assertEqual(payload['normalization']['std'],1.)
            self.assertIn('optimizer_state',payload)
            self.assertIn('rng_states',payload)
            _,_,_,val,_=self.fixtures()
            fixed_time,fixed_noise=validation_draws(val,20,payload['stream_seeds'])
            observed=validation_loss(model,schedule,val,fixed_time,fixed_noise,16)
            self.assertEqual(observed,payload['best_validation_loss'])

    def test_wrong_prediction_shape_cannot_broadcast_into_loss(self):
        class WrongShape(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer=nn.Linear(32,1)
            def forward(self,x,t): return self.layer(x)
        with tempfile.TemporaryDirectory() as directory:
            _,schedule,train,val,config=self.fixtures(steps=1)
            with self.assertRaisesRegex(ValueError,'shape'):
                train_model(WrongShape(),schedule,train,val,config,11,Path(directory)/'best.pt',{'mean':0.,'std':1.})

    def test_missing_or_conflicting_budget_and_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model,schedule,train,val,_=self.fixtures()
            path=Path(directory)/'best.pt'
            for config in (DEFAULT_CONFIG,replace(DEFAULT_CONFIG,training=replace(DEFAULT_CONFIG.training,max_steps=1,epochs=1))):
                with self.assertRaises(ValueError): train_model(model,schedule,train,val,config,11,path,{'mean':0.,'std':1.})
            self.train(path,steps=1)
            with self.assertRaises(FileExistsError): self.train(path,steps=1)


if __name__=='__main__': unittest.main()
