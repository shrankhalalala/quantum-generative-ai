"""Phase 7 protocol serialization and exact batched quantum execution."""
from dataclasses import replace
import json
import unittest
import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
from config import DEFAULT_CONFIG, DiffusionConfig, config_from_dict
from models.quantum_layer import QuantumLayer, BatchedStatevectorEstimator, build_quantum_circuit
from diffusion.noise_schedule import NoiseSchedule


class Phase7Tests(unittest.TestCase):
    def test_budget_selection_uses_validation_gain(self):
        from experiments.phase7 import choose_budget
        history = [{'step':0,'validation_loss':1.}, {'step':500,'validation_loss':.9},
                   {'step':1000,'validation_loss':.89}]
        self.assertEqual(choose_budget(history)['selected_steps'],1000)
        history[-1]['validation_loss'] = .899
        self.assertEqual(choose_budget(history)['selected_steps'],500)

    def test_lambda_selection_enforces_noise_guard_and_tie_rule(self):
        from experiments.phase7 import choose_lambda
        candidates = [{'lambda_acf':weight,'validation_noise_mse':mse,'validation_generated_acf_mae':acf}
                      for weight,mse,acf in ((0.,1.,.4),(.1,1.01,.2),(1.,1.05,.2),(10.,1.2,.1))]
        self.assertEqual(choose_lambda(candidates)['lambda_acf'],.1)
        for row in candidates[1:]: row['validation_noise_mse']=2.
        with self.assertRaises(RuntimeError): choose_lambda(candidates)

    def test_config_json_roundtrip_and_legacy_defaults(self):
        config = replace(DEFAULT_CONFIG,diffusion=DiffusionConfig(steps=100,beta_end=.1),
                         model=replace(DEFAULT_CONFIG.model,quantum_backend='batched_statevector'))
        self.assertEqual(config_from_dict(json.loads(json.dumps(config.to_dict()))),config)
        old = json.loads(json.dumps(DEFAULT_CONFIG.to_dict()))
        del old['model']['quantum_backend']
        del old['loss']
        self.assertEqual(config_from_dict(old),DEFAULT_CONFIG)

    def test_phase7_terminal_signal_variance_is_below_one_percent(self):
        schedule = NoiseSchedule(DiffusionConfig(steps=100,beta_end=.1))
        self.assertLess(float(schedule.alpha_bar[-1]),.01)
        self.assertTrue(torch.isfinite(schedule.posterior_variance).all())
        self.assertTrue((schedule.posterior_variance>=0).all())

    def test_batched_outputs_input_and_weight_gradients_match_qiskit(self):
        for seed in (11,22,33):
            source = torch.randn(3,4,generator=torch.Generator().manual_seed(seed))
            records = []
            for backend in ('qiskit','batched_statevector'):
                layer = QuantumLayer(seed=seed,config=replace(DEFAULT_CONFIG.model,quantum_backend=backend))
                features = source.clone().requires_grad_()
                output = layer(features)
                (output.square().sum()+output[:,0].sum()).backward()
                records.append((output.detach(),features.grad,layer.weight.grad))
            for reference,actual in zip(*records):
                torch.testing.assert_close(actual,reference,atol=2e-7,rtol=2e-6)

    def test_scalar_and_broadcast_pub_expectations_match_qiskit(self):
        spec = build_quantum_circuit()
        values = np.random.default_rng(13).normal(size=(3,20))
        reference,batched = StatevectorEstimator(),BatchedStatevectorEstimator(11)
        for observables,bindings in ((spec.observables[0],values[0]),
                                     ([[obs] for obs in spec.observables],values)):
            pub = (spec.circuit,observables,bindings)
            expected = reference.run([pub]).result()[0].data.evs
            actual = batched.run([pub]).result()[0].data.evs
            np.testing.assert_allclose(actual,expected,atol=2e-14,rtol=2e-14)

    def test_supported_gate_order_and_pauli_products(self):
        circuit = QuantumCircuit(4)
        p = ParameterVector('p',3)
        circuit.rx(p[0],3); circuit.ry(p[1],0); circuit.rz(p[2],2)
        circuit.cx(3,0); circuit.cx(0,1); circuit.cx(1,2)
        observables = [[{'ZZII':.7,'IIZZ':-.2}],['ZIZI']]
        values = np.random.default_rng(9).normal(size=(5,3))
        expected = StatevectorEstimator().run([(circuit,observables,values)]).result()[0].data.evs
        actual = BatchedStatevectorEstimator(11).run([(circuit,observables,values)]).result()[0].data.evs
        np.testing.assert_allclose(actual,expected,atol=2e-14)

    def test_unsupported_circuit_and_observable_fall_back(self):
        for gate,observable in (('h','IIIZ'),('x','IIIX')):
            circuit = QuantumCircuit(4)
            getattr(circuit,gate)(0)
            expected = StatevectorEstimator().run([(circuit,observable)]).result()[0].data.evs
            actual = BatchedStatevectorEstimator(11).run([(circuit,observable)]).result()[0].data.evs
            np.testing.assert_allclose(actual,expected,atol=2e-14)

    def test_batched_training_steps_match_original_qiskit(self):
        from models.factory import make_model
        from training.losses import diffusion_losses
        models = []
        clean = torch.randn(3,32,generator=torch.Generator().manual_seed(55))
        noise = torch.randn(3,32,generator=torch.Generator().manual_seed(56))
        timestep = torch.tensor([0,10,19])
        schedule = NoiseSchedule()
        for backend in ('qiskit','batched_statevector'):
            config = replace(DEFAULT_CONFIG,model=replace(DEFAULT_CONFIG.model,quantum_backend=backend),
                             loss=replace(DEFAULT_CONFIG.loss,objective='noise_acf'))
            model = make_model('hybrid_acf',11,config)
            optimizer = torch.optim.Adam(model.parameters(),lr=.001)
            for _ in range(2):
                optimizer.zero_grad()
                noisy = schedule.add_noise(clean,timestep,noise)
                losses = diffusion_losses(model(noisy,timestep),noise,noisy,clean,timestep,schedule,config.loss)
                losses['total_loss'].backward(); optimizer.step()
            models.append(model)
        for reference,actual in zip(models[0].parameters(),models[1].parameters()):
            torch.testing.assert_close(actual,reference,atol=2e-7,rtol=2e-6)


if __name__ == '__main__': unittest.main()
