# AI Smart Traffic Management (Q-learning)

An adaptive traffic-signal controller for a 4-way intersection, trained with
reinforcement learning and compared against fixed-time and rule-based signals.

## Files
| File | Purpose |
|---|---|
| `traffic_env.py` | Intersection simulator (Poisson arrivals, 1 s steps, min/max green, all-red clearance) |
| `agent.py` | Tabular Q-learning agent |
| `train.py` | Trains on random traffic, saves `q_table.npy` + `training_curve.png` |
| `evaluate.py` | Compares controllers on 4 scenarios, saves `results.png` |

## Run
```
pip install -r requirements.txt
python train.py
python evaluate.py
```

## How it works
- **State:** queue on the green side, queue on the red side (6 buckets each), time in current phase (3 buckets).
- **Actions:** keep the phase or switch (switching is blocked before 5 s green and forced at 40 s).
- **Reward:** `-(total queued cars)/10`, minus 1.5 per switch to avoid flickering.
- **Metric:** average delay per car = queue-seconds / cars departed.

## Extending it
- Emergency-vehicle priority (extra state bit + large reward for clearing its lane)
- Multiple intersections (one agent each, share neighbour queues)
- Deep Q-Network (replace the table with a small neural net)
- Real data: replace Poisson arrivals with counts from a CSV or camera detector (YOLO)
- SUMO/TraCI for realistic vehicle dynamics
