"""Train the Q-learning agent on randomly varying traffic."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from traffic_env import TrafficEnv
from agent import QAgent

EPISODES = 6000
rng = np.random.default_rng(0)
env, agent = TrafficEnv(seed=1), QAgent()
rewards = []
for ep in range(EPISODES):
    eps = max(0.03, 0.4 * (1 - ep / (EPISODES * 0.8)))
    s = env.reset(rng.uniform(0.05, 0.4), rng.uniform(0.05, 0.4))
    total, done = 0.0, False
    while not done:
        a = agent.act(s, eps, rng)
        s2, r, done = env.step(a)
        agent.learn(s, a, r, s2)
        s, total = s2, total + r
    rewards.append(total)
    if (ep + 1) % 1000 == 0:
        print(f"episode {ep+1}/{EPISODES}  avg reward (last 1000): {np.mean(rewards[-1000:]):.1f}")
agent.save("q_table.npy")
k = 100
plt.figure(figsize=(6, 3.5))
plt.plot(np.convolve(rewards, np.ones(k) / k, mode="valid"))
plt.xlabel("episode"); plt.ylabel("reward (smoothed)"); plt.title("Training curve")
plt.tight_layout(); plt.savefig("training_curve.png", dpi=130)
print("saved q_table.npy and training_curve.png")
