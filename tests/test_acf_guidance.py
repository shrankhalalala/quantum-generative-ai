"""Mathematical ACF checks, quantum gradients, zero-weight ablation and metrics."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from config import DEFAULT_CONFIG, LossConfig
from diffusion.noise_schedule import NoiseSchedule
from evaluation.metrics import evaluate_samples
from models.factory import make_model, model_state_sha256
from training.losses import sequence_acf, reconstruct_clean, autocorrelation_loss, diffusion_losses
from training.train import train_model, restore_checkpoint
from utils.time_series import sequence_acf as numpy_acf


class ACFGuidanceTests(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(42)
        self.clean = torch.randn(2,32,generator=self.generator)
        self.noise = torch.randn(2,32,generator=self.generator)
        self.time = torch.tensor([0,19])
        self.schedule = NoiseSchedule()
        self.noisy = self.schedule.add_noise(self.clean,self.time,self.noise)

    def test_acf_matches_existing_numpy_definition(self):
        actual = sequence_acf(self.clean.double()).numpy()
        np.testing.assert_allclose(actual,numpy_acf(self.clean.double().numpy())[:,1:],atol=1e-14)

    def test_reconstruction_with_perfect_noise_prediction(self):
        recovered = reconstruct_clean(self.noisy,self.noise,self.time,self.schedule)
        torch.testing.assert_close(recovered,self.clean,atol=3e-7,rtol=1e-6)
        self.assertLess(float(autocorrelation_loss(recovered,self.clean)),1e-12)

    def test_acf_gradcheck(self):
        values = self.clean.double().requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda x: sequence_acf(x), (values,)))

    def test_constant_and_near_constant_signals_have_finite_gradients(self):
        for scale in (0.,1e-9):
            values = (torch.ones(2,32)+scale*self.clean).requires_grad_()
            loss = autocorrelation_loss(values,self.clean)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(values.grad).all())

    def test_acf_loss_reaches_quantum_and_classical_parameters(self):
        model = make_model('hybrid_acf',11)
        prediction = model(self.noisy,self.time)
        recovered = reconstruct_clean(self.noisy,prediction,self.time,self.schedule)
        autocorrelation_loss(recovered,self.clean).backward()
        for name in ('encoder','bottleneck','decoder','timestep_encoding'):
            gradients = [p.grad for p in getattr(model,name).parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.square().sum()) for g in gradients),0.)

    def test_guided_and_unguided_hybrids_start_identically(self):
        for seed in DEFAULT_CONFIG.training.seeds:
            self.assertEqual(model_state_sha256(make_model('hybrid',seed)),
                             model_state_sha256(make_model('hybrid_acf',seed)))

    def test_positive_guidance_changes_objective_and_gradient(self):
        prediction = torch.zeros_like(self.noise,requires_grad=True)
        losses = diffusion_losses(prediction,self.noise,self.noisy,self.clean,self.time,
                                  self.schedule,LossConfig(objective='noise_acf',lambda_acf=2.))
        torch.testing.assert_close(losses['total_loss'],losses['noise_loss']+2*losses['acf_loss'])
        noise_gradient = torch.autograd.grad(losses['noise_loss'],prediction,retain_graph=True)[0]
        total_gradient = torch.autograd.grad(losses['total_loss'],prediction)[0]
        self.assertGreater(float((total_gradient-noise_gradient).norm()),0.)

    def test_zero_weight_matches_hybrid_training_and_checkpoint_restores(self):
        base = replace(DEFAULT_CONFIG,training=replace(DEFAULT_CONFIG.training,max_steps=2,batch_size=2))
        guided = replace(base,loss=LossConfig(objective='noise_acf',lambda_acf=0.))
        models = [make_model('hybrid',11),make_model('hybrid_acf',11)]
        with tempfile.TemporaryDirectory() as directory:
            results = []
            for index,(model,config) in enumerate(zip(models,(base,guided))):
                result = train_model(model,NoiseSchedule(),self.clean,self.clean,config,11,
                                     Path(directory)/f'{index}.pt',{},1)
                results.append(result)
            self.assertEqual(model_state_sha256(models[0]),model_state_sha256(models[1]))
            self.assertEqual(results[0].history,results[1].history)
            clone = make_model('hybrid_acf',22)
            payload = restore_checkpoint(clone,NoiseSchedule(),results[1].checkpoint)
            self.assertEqual(payload['config']['loss']['objective'],'noise_acf')
            self.assertEqual(payload['config']['loss']['lambda_acf'],0.)
            self.assertEqual(payload['checkpoint_selection'],'validation_noise_mse')
            self.assertEqual(payload['step'],results[1].summary['best_step'])
            expected = models[1]
            restore_checkpoint(expected,NoiseSchedule(),results[1].checkpoint)
            self.assertEqual(model_state_sha256(clone),model_state_sha256(expected))

    def test_invalid_loss_configurations_rejected(self):
        for loss in (LossConfig(objective='unknown'),LossConfig(lambda_acf=-1.),
                     LossConfig(lambda_acf=float('nan')),LossConfig(lambda_acf=float('inf')),
                     LossConfig(max_lag=0),LossConfig(max_lag=32),LossConfig(epsilon=0.)):
            with self.subTest(loss=loss),self.assertRaises(ValueError):
                replace(DEFAULT_CONFIG,loss=loss).validate()

    def test_metrics_identity_and_known_offset(self):
        reference = self.clean.double().numpy()
        same = evaluate_samples(reference,reference)
        self.assertEqual(same['marginal_wasserstein_1'],0.)
        self.assertEqual(same['acf_mae'],0.)
        shifted = evaluate_samples(reference+3.,reference)
        self.assertAlmostEqual(shifted['marginal_wasserstein_1'],3.)
        self.assertAlmostEqual(shifted['mean']['difference'],3.)
        self.assertAlmostEqual(shifted['std']['absolute_error'],0.)
        self.assertAlmostEqual(shifted['acf_mae'],0.)

    def test_metrics_flag_constant_rows_without_dropping_them(self):
        result = evaluate_samples(np.ones((2,32)),self.clean.numpy())
        self.assertIsNone(result['acf_mae'])
        self.assertEqual(result['near_constant_rows']['generated'],[0,1])
        self.assertEqual(result['n_generated'],2)
        self.assertTrue(np.isfinite(result['marginal_wasserstein_1']))

    def test_invalid_metric_and_loss_inputs(self):
        with self.assertRaises(ValueError): evaluate_samples(np.ones((2,31)),np.ones((2,32)))
        with self.assertRaises(ValueError): sequence_acf(self.clean,32)
        with self.assertRaises(ValueError): sequence_acf(self.clean,epsilon=0.)
        with self.assertRaises(ValueError): autocorrelation_loss(self.clean,self.clean[:1])
        with self.assertRaises(ValueError): reconstruct_clean(self.noisy,self.noise[:1],self.time,self.schedule)


if __name__ == '__main__':
    unittest.main()
