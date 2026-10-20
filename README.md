# TD-error-seeking exploration: baseline vs. stuck-transition-capped variant

## What this does
`analysis.py` compares two tabular Q-learning agents on a simulated 5x5 stochastic-reward gridworld (sparse goal, noisy costly cells). Both choose actions by argmax of a running mean |TD error| per state-action pair, with eps=0.05 exploration.

- **Baseline:** plain error-seeking rule.
- **Improved:** pairs whose |TD error| history stays high without decaying are flagged as "stuck". Their priority is capped at a visited-value quantile. The quantile (0, 0.25 or 0.5) is chosen by leave-one-training-env-out cross-validation on the training seeds. The test seeds are never used for selection.

The main score is the mean number of environment steps to the first goal on 20 held-out evaluation seeds, over 3 agent seeds. Lower is better.

## Run
```
pip install -r requirements.txt
python analysis.py
```
Outputs: `results.json` and `figures/steps_to_goal.png`. No downloads are needed. Seed is 42.

## Notes
- Simulated toy environment. This is a stand-in for SEE, not SAC with neural networks.
- Only three agent seeds, and no significance test.
- The earlier median-only cap lost to the baseline (208.1 vs 136.5 steps), so the cap level was made a CV-selected choice.

## Reference

This project is motivated by the following paper:

> S Griesbach, C D'Eramo (2025). *Learning to Explore in Diverse Reward Settings via Temporal-Difference-Error Maximization*. arXiv preprint arXiv:2506.13345. https://arxiv.org/abs/2506.13345

Paper author: Prof. Carlo D'Eramo, Julius-Maximilians-Universität Würzburg (carlo.deramo@uni-wuerzburg.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
