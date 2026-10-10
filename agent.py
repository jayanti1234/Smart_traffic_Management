import numpy as np
from traffic_env import TrafficEnv


class QAgent:
    def __init__(self, alpha=0.1, gamma=0.95):
        self.Q = np.zeros((TrafficEnv.n_states, TrafficEnv.n_actions))
        self.alpha, self.gamma = alpha, gamma

    def act(self, s, eps=0.0, rng=np.random):
        if rng.random() < eps:
            return rng.integers(2)
        q = self.Q[s]
        return rng.integers(2) if q[0] == q[1] else int(q.argmax())

    def learn(self, s, a, r, s2):
        target = r + self.gamma * self.Q[s2].max()
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def save(self, path="q_table.npy"):
        np.save(path, self.Q)

    def load(self, path="q_table.npy"):
        self.Q = np.load(path)
