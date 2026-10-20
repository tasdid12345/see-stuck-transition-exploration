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

## Results (latest run)

On a simulated stochastic-reward gridworld (held-out evaluation seeds, mean environment steps to first goal, lower is better), a tabular TD-error-seeking agent averaged 136.45 steps while my capped variant averaged 129.57, a small (about 5%) gain within the seed-to-seed spread (std about 31–35).

| Metric | Value |
|---|---|
| baseline_score | 136.5 |
| improved_score | 129.6 |
| improvement | 6.883 |
| baseline_score_std | 31.47 |
| improved_score_std | 34.73 |
| baseline_train_steps | 113.8 |
| improved_train_steps | 107.3 |
| baseline_stuck_frac | 0.2573 |
| improved_stuck_frac | 0.04026 |
| baseline_greedy_return | 4.386 |
| improved_greedy_return | 6.262 |
| improved_stuck_threshold | 0.25 |
| improved_cap_quantile | 0 |
| improved_cap_none_selected | 0 |
| cv_train_steps_none | 113.8 |
| cv_train_steps_thr0.25_q0.0 | 107.3 |
| cv_train_steps_thr0.25_q0.25 | 107.3 |
| cv_train_steps_thr0.25_q0.5 | 107.3 |
| cv_train_steps_thr0.5_q0.0 | 107.3 |
| cv_train_steps_thr0.5_q0.25 | 107.3 |
| cv_train_steps_thr0.5_q0.5 | 107.3 |
| cv_train_steps_thr1.0_q0.0 | 107.3 |
| cv_train_steps_thr1.0_q0.25 | 107.3 |
| cv_train_steps_thr1.0_q0.5 | 107.3 |

Run on 2026-10-20. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Simulated toy environment; tabular Q-learning, not SAC with neural networks, so this is only a stand-in for SEE. Only the improved variant's stuck threshold and cap were tuned, by CV on TRAIN_ENV; the baseline and the test seeds were not used for selection. The candidate set includes the no-cap rule, so the selected setting is no worse than the baseline on training seeds; test results can still differ. Steps are censored at the step budget. Only three agent seeds and three training envs make the CV noisy; no significance test.

## Reference

This project is motivated by the following paper:

> S Griesbach, C D'Eramo (2025). *Learning to Explore in Diverse Reward Settings via Temporal-Difference-Error Maximization*. arXiv preprint arXiv:2506.13345. https://arxiv.org/abs/2506.13345

Paper author: Prof. Carlo D'Eramo, Julius-Maximilians-Universität Würzburg (carlo.deramo@uni-wuerzburg.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
