"""Compare fixed-time, rule-based and AI controllers on the same traffic."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from traffic_env import TrafficEnv, MIN_GREEN
from agent import QAgent

agent = QAgent(); agent.load("q_table.npy")

def fixed(cycle):
    return lambda env, s: int(env.green_t >= cycle)

def rule(env, s):  # switch when the red side is clearly longer
    p = env.phase
    g, r = env.q[p] + env.q[p + 2], env.q[1 - p] + env.q[3 - p]
    return int(env.green_t >= MIN_GREEN and r > g + 2)

def ai(env, s):
    return int(agent.Q[s].argmax())

controllers = {"Fixed 15s": fixed(15), "Fixed 30s": fixed(30),
               "Rule-based": rule, "AI (Q-learning)": ai}
scenarios = {"Balanced": (0.2, 0.2), "Rush hour N-S": (0.35, 0.08),
             "Rush hour E-W": (0.08, 0.35), "Heavy": (0.38, 0.38)}

results = {c: [] for c in controllers}
print(f"{'Scenario':16}" + "".join(f"{c:>17}" for c in controllers))
for name, (ns, ew) in scenarios.items():
    row = []
    for cname, ctrl in controllers.items():
        delays = []
        for seed in range(20):  # identical seeds for every controller
            env = TrafficEnv(seed=100 + seed, horizon=1800)
            s, done = env.reset(ns, ew), False
            while not done:
                s, _, done = env.step(ctrl(env, s))
            delays.append(env.avg_delay)
        results[cname].append(np.mean(delays)); row.append(np.mean(delays))
    print(f"{name:16}" + "".join(f"{v:17.1f}" for v in row))

x = np.arange(len(scenarios)); w = 0.2
plt.figure(figsize=(8, 4))
for i, (c, v) in enumerate(results.items()):
    plt.bar(x + i * w, v, w, label=c)
plt.xticks(x + 1.5 * w, scenarios.keys()); plt.ylabel("avg delay (s per car)")
plt.title("Average delay by controller (lower is better)"); plt.legend()
plt.tight_layout(); plt.savefig("results.png", dpi=130)
print("saved results.png")
