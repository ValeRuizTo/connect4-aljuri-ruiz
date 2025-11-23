import numpy as np
from connect4.policy import Policy


class HelloPolicy(Policy):

    def mount(self) -> None:
        pass

    def act(self, s: np.ndarray) -> int:
        rng = np.random.default_rng()
        available_cols = [c for c in range(7) if s[0, c] == 0]
        return int(rng.choice(available_cols))
    
    def learn(self, reward):
    # Los agentes random NO aprenden, pero necesitamos esta función
    # para que training_env no falle.
        pass
