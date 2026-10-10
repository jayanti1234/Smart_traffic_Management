"""Single 4-way intersection, 1-second time steps.
Approaches: 0=North, 1=East, 2=South, 3=West.
Phase 0 = North-South green, phase 1 = East-West green.
"""
import numpy as np

SAT_FLOW = 1      # cars per second that can leave a green approach
MIN_GREEN = 5     # seconds
MAX_GREEN = 40    # seconds
ALL_RED = 2       # clearance time after every switch


def bucket(q):
    for i, t in enumerate((1, 3, 6, 11, 19)):
        if q < t:
            return i
    return 5


class TrafficEnv:
    n_states = 6 * 6 * 3
    n_actions = 2  # 0 = keep phase, 1 = switch phase

    def __init__(self, ns_rate=0.25, ew_rate=0.25, horizon=600, seed=None):
        self.rates = np.array([ns_rate, ew_rate, ns_rate, ew_rate])
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)

    def reset(self, ns_rate=None, ew_rate=None):
        if ns_rate is not None:
            self.rates = np.array([ns_rate, ew_rate, ns_rate, ew_rate])
        self.q = np.zeros(4, dtype=int)
        self.phase, self.green_t, self.clear_t, self.t = 0, 0, 0, 0
        self.queue_seconds, self.departed = 0.0, 0
        return self.state()

    def state(self):
        p = self.phase
        g = self.q[p] + self.q[p + 2]
        r = self.q[1 - p] + self.q[3 - p]
        t = 0 if self.green_t < 10 else 1 if self.green_t < 20 else 2
        return (bucket(g) * 6 + bucket(r)) * 3 + t

    def step(self, action):
        switched = False
        if self.clear_t == 0:  # free to decide
            if self.green_t >= MAX_GREEN:
                action = 1
            elif self.green_t < MIN_GREEN:
                action = 0
            if action == 1:
                self.clear_t, switched = ALL_RED, True
        # signal timing
        if self.clear_t > 0:
            self.clear_t -= 1
            if self.clear_t == 0:
                self.phase ^= 1
                self.green_t = 0
        else:
            self.green_t += 1
            for d in (self.phase, self.phase + 2):  # discharge
                n = min(self.q[d], SAT_FLOW)
                self.q[d] -= n
                self.departed += n
        self.q += self.rng.poisson(self.rates)  # arrivals
        total = int(self.q.sum())
        self.queue_seconds += total
        self.t += 1
        reward = -total / 10.0 - (1.5 if switched else 0.0)
        return self.state(), reward, self.t >= self.horizon

    @property
    def avg_delay(self):
        """Average seconds each car spends in queue (Little's law)."""
        return self.queue_seconds / max(self.departed, 1)
