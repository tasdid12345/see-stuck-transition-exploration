# Stuck-Transition-Capped TD-Error Exploration on a Stochastic Gridworld

Motivation: "Learning to Explore in Diverse Reward Settings via Temporal-Difference-Error Maximization" (2025) proposes Stable Errorseeking Exploration (SEE), which drives an agent toward transitions with large TD errors and combines this with Soft Actor-Critic. Error-seeking can get stuck on transitions whose TD error stays high because of reward noise rather than because they are informative. This experiment asks whether detecting and capping such transitions helps.

Question: Can a TD-error-seeking exploration rule that detects and caps persistently high-TD-error transitions reach the sparse-reward goal faster than the plain TD-error-seeking rule in a small stochastic-reward gridworld?

Dataset: a small stochastic-reward gridworld generated in numpy inside the script (sparse goal reward, slip noise, and a few costly cells with noisy rewards). This is not the paper's own dataset (same_as_paper is false), and nothing is downloaded.

Baseline vs. improved: the baseline is tabular Q-learning that picks actions by maximizing a running estimate of the absolute TD error per state-action pair, a stand-in for SEE's error-seeking objective. The improved agent keeps a per-pair TD-error history and caps the exploration value of pairs whose TD error stays high without decaying. Both share the same environment, data loading, budget and evaluation code.

How to run:

    pip install -r requirements.txt
    python analysis.py

Results: results are written to results.json (and a chart to figures/steps_to_goal.png).

## Reference

This project is motivated by the following paper:

> S Griesbach, C D'Eramo (2025). *Learning to Explore in Diverse Reward Settings via Temporal-Difference-Error Maximization*. arXiv preprint arXiv:2506.13345. https://arxiv.org/abs/2506.13345

Paper author: Prof. Carlo D'Eramo, Julius-Maximilians-Universität Würzburg (carlo.deramo@uni-wuerzburg.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
