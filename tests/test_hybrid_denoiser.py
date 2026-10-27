"""Only-bottleneck replacement, paired initialization and diffusion integration."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from config import DEFAULT_CONFIG
from models.classical_denoiser import ClassicalDenoiser, trainable_parameters
from models.factory import make_model, copy_shared_state, shared_state_sha256, SHARED_COMPONENTS
from models.hybrid_denoiser import HybridDenoiser
from models.quantum_layer import QuantumLayer
from diffusion.noise_schedule import NoiseSchedule
from diffusion.sampling import sample
from training.train import train_model, restore_checkpoint


class HybridDenoiserTests(unittest.TestCase):
    def test_counts_and_only_bottleneck_replacement(self):
        models={name:make_model(name,11) for name in ('classical','hybrid','classical_matched')}
        self.assertEqual(trainable_parameters(models['classical']),1816)
        self.assertEqual(trainable_parameters(models['hybrid']),1812)
        self.assertEqual(trainable_parameters(models['classical_matched']),1812)
        self.assertIsInstance(models['hybrid'].bottleneck,QuantumLayer)
        self.assertIsNone(models['classical_matched'].bottleneck[0].bias)
        self.assertEqual(len(set(shared_state_sha256(model) for model in models.values())),1)
        torch.testing.assert_close(models['classical'].bottleneck[0].weight,
                                   models['classical_matched'].bottleneck[0].weight,atol=0,rtol=0)
        for component in SHARED_COMPONENTS:
            baseline=getattr(models['classical'],component)
            for name in ('hybrid','classical_matched'):
                other=getattr(models[name],component)
                self.assertEqual(type(baseline),type(other))
                for key,value in baseline.state_dict().items():
                    torch.testing.assert_close(value,other.state_dict()[key],atol=0,rtol=0)
                    self.assertNotEqual(value.data_ptr(),other.state_dict()[key].data_ptr())

    def test_pairing_for_each_repetition_and_no_global_rng_consumption(self):
        for seed in DEFAULT_CONFIG.training.seeds:
            np.random.seed(41)
            expected_numpy=np.random.rand(3)
            np.random.seed(41)
            torch.manual_seed(41)
            before=torch.get_rng_state().clone()
            a=make_model('classical',seed)
            b=make_model('hybrid',seed)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            np.testing.assert_array_equal(np.random.rand(3),expected_numpy)
            self.assertEqual(shared_state_sha256(a),shared_state_sha256(b))
        self.assertNotEqual(shared_state_sha256(make_model('classical',11)),
                            shared_state_sha256(make_model('classical',22)))

    def test_factory_preserves_original_classical_initialization(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(11)
            original=ClassicalDenoiser()
        created=make_model('classical',11)
        for key,value in original.state_dict().items():
            torch.testing.assert_close(value,created.state_dict()[key],atol=0,rtol=0)

    def test_full_diffusion_gradients_and_output_shape(self):
        model=make_model('hybrid',11)
        rng=torch.Generator().manual_seed(8)
        clean=torch.randn(2,32,generator=rng,requires_grad=True)
        time=torch.tensor([0,19],dtype=torch.long)
        noise=torch.randn(2,32,generator=rng)
        output=model(NoiseSchedule().add_noise(clean,time,noise),time)
        self.assertEqual(output.shape,(2,32))
        loss=(output-noise).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(clean.grad).all())
        self.assertGreater(float(clean.grad.norm()),0.)
        for component in (*SHARED_COMPONENTS,'bottleneck'):
            gradients=[p.grad for p in getattr(model,component).parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.double().square().sum()) for g in gradients),0.)

    def test_same_trainer_updates_classical_and_quantum_and_restores_checkpoint(self):
        model=make_model('hybrid',11)
        schedule=NoiseSchedule()
        rng=torch.Generator().manual_seed(3)
        train=torch.randn(2,32,generator=rng)
        validation=torch.randn(2,32,generator=rng)
        config=replace(DEFAULT_CONFIG,training=replace(DEFAULT_CONFIG.training,max_steps=1))
        before={key:value.detach().clone() for key,value in model.named_parameters()}
        with tempfile.TemporaryDirectory() as directory:
            result=train_model(model,schedule,train,validation,config,11,Path(directory)/'best.pt',
                               {'mean':0.,'std':1.,'n_sequences':2,'sequence_length':32,'ddof':0},1)
            self.assertEqual(result.summary['steps'],1)
            self.assertEqual(result.summary['examples_seen'],2)
            self.assertEqual(len(result.summary['gradient_norms_per_update']),1)
            self.assertEqual(result.summary['quantum_submitted_workloads']['observable_expectation_evaluations'],344)
            for component in (*SHARED_COMPONENTS,'bottleneck'):
                self.assertGreater(result.summary['gradient_norms_per_update'][0][component],0.)
                self.assertGreater(result.summary['last_model_parameter_change_l2'][component],0.)
                self.assertTrue(any(not torch.equal(p,before[name]) for name,p in model.named_parameters()
                                    if name.startswith(component+'.')))
            payload=restore_checkpoint(model,schedule,result.checkpoint)
            self.assertEqual(payload['step'],result.summary['best_step'])
            self.assertEqual(payload['validation_every'],1)
            clone=make_model('hybrid',22)
            restore_checkpoint(clone,NoiseSchedule(),result.checkpoint)
            with torch.no_grad():
                torch.testing.assert_close(model(train,torch.tensor([0,19])),clone(train,torch.tensor([0,19])),atol=0,rtol=0)
            with self.assertRaisesRegex(ValueError,'model type'):
                restore_checkpoint(make_model('classical',11),NoiseSchedule(),result.checkpoint)

    def test_hybrid_sampling_shape_repeatability_and_no_gradients(self):
        model=make_model('hybrid',11)
        schedule=NoiseSchedule()
        first=sample(model,schedule,2,10011,batch_size=2)
        second=sample(model,schedule,2,10011,batch_size=2)
        self.assertEqual(first.samples.shape,(2,32))
        self.assertTrue(torch.isfinite(first.samples).all())
        self.assertFalse(first.samples.requires_grad)
        self.assertEqual(len(first.step_max_abs),20)
        self.assertTrue(model.training)
        torch.testing.assert_close(first.samples,second.samples,atol=0,rtol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_copy_shared_weights_does_not_change_bottleneck_or_alias_parameters(self):
        source=make_model('classical',11)
        target=make_model('hybrid',22)
        original=target.bottleneck.weight.detach().clone()
        copy_shared_state(source,target)
        self.assertEqual(shared_state_sha256(source),shared_state_sha256(target))
        torch.testing.assert_close(original,target.bottleneck.weight,atol=0,rtol=0)
        with torch.no_grad():
            target.encoder[0].weight.add_(1.)
        self.assertNotEqual(shared_state_sha256(source),shared_state_sha256(target))

    def test_bad_inputs_and_factory_settings_rejected(self):
        model=HybridDenoiser()
        for values,time in ((torch.zeros(2,31),torch.tensor([0,0])),
                            (torch.zeros(2,32),torch.tensor([0.,0.])),
                            (torch.full((2,32),float('nan')),torch.tensor([0,0]))):
            with self.assertRaises(ValueError): model(values,time)
        with self.assertRaises(ValueError): make_model('missing')
        for seed in (-1,2**32,1.5):
            with self.assertRaises(ValueError): make_model('hybrid',seed)


if __name__=='__main__': unittest.main()
