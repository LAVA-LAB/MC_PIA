import numpy as np
class AdversarialAttack:

    def __init__(self, state_mapper, attack_config_str:str) -> None:
        """
        Initialize the attack.
        :param state_mapper: The state mapper.
        :param attack_config: The attack configuration.
        """
        self.state_mapper = state_mapper
        self.attack_config_str = attack_config_str
        self.attack_buffer = {}
        self.max_l1_norm = 0

    def parse_attack_config(self, attack_config_str:str) -> None:
        """
        Parse the attack configuration.
        :param attack_config_str: The attack configuration.
        """
        raise NotImplementedError()

    def attack(self, rl_agent, state:np.ndarray) -> np.ndarray:
        """
        Perform the attack.
        :param state: The state.
        :return: The adversarial state.
        """
        raise NotImplementedError()

    def already_attacked(self, state: np.ndarray) -> bool:
        if str(state) in self.attack_buffer.keys():
            return True
        return False
    
    def update_attack_buffer(self, state, attack):
        self.attack_buffer = {}
        self.attack_buffer[str(state)] = attack

    def save_max_l1_norm(self, attack):
        l1_norm = np.sum(np.abs(attack))
        if l1_norm > self.max_l1_norm:
            self.max_l1_norm = l1_norm