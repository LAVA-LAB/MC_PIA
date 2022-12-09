from common.adversarial_attacks.adversarial_attack import AdversarialAttack
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import os

class FeatureFGSM(AdversarialAttack):

    def __init__(self, state_mapper, attack_config_str: str) -> None:
        super().__init__(state_mapper, attack_config_str)
        self.parse_attack_config(attack_config_str)

    def parse_attack_config(self, attack_config_str: str) -> None:
        attack_name, epsilon, direction = attack_config_str.split(',')
        self.attack_name = attack_name
        self.epsilon = float(epsilon)
        self.direction = self.state_mapper.mapper[str(direction)]


    def attack(self, rl_agent, state: np.ndarray) -> np.ndarray:
        if self.already_attacked(state):
            return state + self.attack_buffer[str(state)]
        else:
            # Numpy array to torch array with requires_grad=True
            state = torch.from_numpy(state).float().requires_grad_(True)
            output = rl_agent.q_eval(state).reshape(1, -1)
            target = output.argmax().reshape(-1)
            loss = F.nll_loss(output, target)
            rl_agent.q_eval.zero_grad()
            loss.backward()
            data_grad = state.grad.data
            sign_data_grad = data_grad.sign()
            # Create torch array with 0s of size of sign_data_grad
            adv_perturbation = torch.zeros(sign_data_grad.size())
            adv_perturbation[self.direction] = sign_data_grad[self.direction]
            adv_perturbation = adv_perturbation.numpy()
            state = state.detach().numpy()
            adversarial_state = state + adv_perturbation * self.epsilon
            self.update_attack_buffer(state, adv_perturbation)
            self.save_max_l1_norm(adv_perturbation)
            return adversarial_state