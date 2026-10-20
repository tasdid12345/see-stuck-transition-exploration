# TD-error-seeking exploration: stuck-transition cap

Question: can a TD-error-seeking exploration rule that detects and caps persistently high-TD-error
transitions reach the sparse-reward goal faster than the plain TD-error-seeking rule in a small
stochastic-reward gridworld?

Run: `pip install -r requirements.txt && python analysis.py`. Writes `results.json` and
`figures/steps_to_goal.png`. No downloads; CPU only; seed 42.

Method (changed): the improved variant's stuck threshold (0.25, 0.5, 1.0) and cap quantile
(none, 0, 0.25, 0.5) are chosen by leave-one-training-env-out CV on the training seeds only.
The "none" candidate is identical to the baseline rule, so the selected setting is no worse than
the baseline on training seeds. The baseline and the held-out test seeds (1000–1019) are not used
for selection. The metric is mean steps to first goal (lower is better), with three agent seeds.

## Reference

This project is motivated by the following paper:

> S Griesbach, C D'Eramo (2025). *Learning to Explore in Diverse Reward Settings via Temporal-Difference-Error Maximization*. arXiv preprint arXiv:2506.13345. https://arxiv.org/abs/2506.13345

Paper author: Prof. Carlo D'Eramo, Julius-Maximilians-Universität Würzburg (carlo.deramo@uni-wuerzburg.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
