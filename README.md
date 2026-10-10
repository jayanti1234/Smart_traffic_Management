# AI Smart Traffic Management (Q-learning)

Most traffic lights run on a fixed timer. They give 30 seconds to one road, then 30 seconds to the other, even if one road is empty and the other has a long line of cars. This project tries a different approach: a small AI agent that watches the queues at a four-way intersection and decides, second by second, whether to keep the current green or switch.

The agent is trained with Q-learning, a basic reinforcement learning method. It learns purely from trial and error in a simulation, with no hand-written timing rules.

How it works

The intersection has four approaches (north, east, south, west) and two signal phases: north-south green or east-west green. Every simulated second the agent looks at the situation and picks one of two actions: keep the current phase, or switch.

What the agent sees (the state):

how many cars are waiting on the road that currently has green
how many are waiting on the roads with red
how long the current green has been on

The queue sizes are grouped into six rough buckets (empty, 1-2, 3-5, 6-10, 11-18, 19+) so the table stays small enough to learn quickly.

What it is rewarded for: fewer waiting cars. Each second costs a penalty proportional to the total queue length, plus a small extra penalty every time it switches, so the light does not flicker back and forth.

Safety rules built into the simulator: a green phase lasts at least 5 seconds and at most 40 seconds, and every switch is followed by 2 seconds of all-red so the intersection can clear. The agent cannot override these.

During training the agent plays thousands of short episodes with randomly changing traffic levels, so it does not just memorize one situation.

Results

I compared four controllers on the same random traffic (20 runs of 30 simulated minutes each, same seeds for everyone). The number is the average time a car spends waiting in the queue, in seconds. Lower is better.
