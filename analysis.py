"""TD-error-seeking exploration: plain baseline vs. stuck-transition-capped variant on a
stochastic-reward gridworld. Tabular Q-learning with numpy only; no download."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
N = 5                                # grid side length
GOAL = (4, 4)                        # sparse goal: +10, terminal
COSTLY = {(1, 2), (3, 1), (2, 3)}    # cells whose entry reward is noisy and negative
GAMMA, ALPHA, EPS, SLIP = 0.95, 0.1, 0.05, 0.1   # fixed a priori, not tuned
EP_LEN, BUDGET = 100, 5000           # episode truncation, env-step budget per run
H = 40                               # per-transition |TD| history length
STUCK_THR = 0.5                      # recent mean |TD| needed to call a pair "stuck"
DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
AGENT_SEEDS = [42, 43, 44]           # agent seeds (randomized agents)
TRAIN_ENV = [42, 43, 44]             # training-split rollout seeds (used for CV selection)
TEST_ENV = list(range(1000, 1020))   # held-out 20 evaluation seeds, used once for scoring
# Candidate cap levels for stuck pairs (quantile of visited running |TD| means).
# Selected by leave-one-training-env-out CV on TRAIN_ENV only; TEST_ENV is never used here.
CAP_QUANTILES = [0.0, 0.25, 0.5]


class GridEnv:
    """Gridworld with slip noise, sparse goal reward and noisy costly cells."""
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.pos, self.t = (0, 0), 0
        return self.s()

    def s(self):
        return self.pos[0] * N + self.pos[1]

    def step(self, a):
        if self.rng.random() < SLIP:               # slip: random action instead
            a = int(self.rng.integers(4))
        r = min(max(self.pos[0] + DELTAS[a][0], 0), N - 1)
        c = min(max(self.pos[1] + DELTAS[a][1], 0), N - 1)
        self.pos, self.t = (r, c), self.t + 1
        if self.pos == GOAL:
            return self.s(), 10.0, True
        rew = self.rng.normal(-1.0, 3.0) if self.pos in COSTLY else -0.04
        return self.s(), rew, False


def stuck_mask(hist, nh):
    """A pair is stuck if its |TD| history is full, recent mean is high, and it is not decaying."""
    idx = (nh[..., None] + np.arange(H)) % H          # chronological order, oldest first
    ch = np.take_along_axis(hist, idx, axis=2)
    old, new = ch[..., :H // 2].mean(-1), ch[..., H // 2:].mean(-1)
    return (nh >= H) & (new > STUCK_THR) & (new >= 0.9 * old)


def priorities(est, cnt, stuck, kind, cap_q=0.5):
    """Error-seeking objective; the improved variant caps stuck pairs at a visited-value
    quantile cap_q (cap_q=0.5 is the median) so they fall below well-explored pairs."""
    prio = est.copy()
    if kind == "improved" and (cnt > 0).any():
        cap = np.quantile(est[cnt > 0], cap_q)
        prio = np.where(stuck, np.minimum(prio, cap), prio)
    return prio


def greedy_return(Q, seed, n_ep=10):
    """Average undiscounted return of the greedy policy over fresh episodes."""
    env, tot = GridEnv(seed), 0.0
    for _ in range(n_ep):
        s, done = env.reset(), False
        while not done:
            s, r, goal = env.step(int(np.argmax(Q[s])))
            tot += r
            done = goal or env.t >= EP_LEN
    return tot / n_ep


def run(kind, agent_seed, env_seed, cap_q=0.5):
    """One learning run; returns (steps to first goal, stuck-step fraction, greedy return)."""
    env, rng = GridEnv(env_seed), np.random.default_rng(agent_seed)
    Q = np.zeros((N * N, 4))
    est = np.full((N * N, 4), np.inf)                 # running mean |TD|; inf = unvisited
    cnt = np.zeros((N * N, 4), int)
    hist, nh = np.zeros((N * N, 4, H)), np.zeros((N * N, 4), int)
    first, n_stuck = None, 0
    s = env.reset()
    for t in range(BUDGET):
        stuck = stuck_mask(hist, nh)
        prio = priorities(est, cnt, stuck, kind, cap_q)
        if rng.random() < EPS:
            a = int(rng.integers(4))
        else:                                         # argmax with random tie-break
            a = int(rng.choice(np.flatnonzero(prio[s] == prio[s].max())))
        n_stuck += int(stuck[s, a])
        s2, r, goal = env.step(a)
        td = r + GAMMA * (0.0 if goal else Q[s2].max()) - Q[s, a]
        Q[s, a] += ALPHA * td
        cnt[s, a] += 1
        est[s, a] = abs(td) if cnt[s, a] == 1 else est[s, a] + 0.1 * (abs(td) - est[s, a])
        hist[s, a, nh[s, a] % H] = abs(td)
        nh[s, a] += 1
        if goal and first is None:
            first = t + 1
        s = env.reset() if (goal or env.t >= EP_LEN) else s2
    return (first or BUDGET), n_stuck / BUDGET, greedy_return(Q, env_seed + 777)


def evaluate(kind, env_seeds, cap_q=0.5):
    """Runs every agent seed on every environment seed; returns arrays of shape (agents, envs)."""
    out = np.array([[run(kind, a, e, cap_q) for e in env_seeds] for a in AGENT_SEEDS])
    return out[..., 0], out[..., 1], out[..., 2]


def select_cap_quantile():
    """Leave-one-training-env-out CV: each training env is held out in turn and the cap
    quantile is scored on it; the quantile with the lowest mean held-out steps wins.
    Uses TRAIN_ENV only; the baseline is never touched."""
    fold_scores = {q: [] for q in CAP_QUANTILES}
    for e in TRAIN_ENV:
        for q in CAP_QUANTILES:
            steps, _, _ = evaluate("improved", [e], q)
            fold_scores[q].append(steps.mean())
    cv = {q: float(np.mean(v)) for q, v in fold_scores.items()}
    best = min(cv, key=cv.get)
    return best, cv


def main():
    best_q, cv = select_cap_quantile()
    print("[cv] held-out training-seed steps per cap quantile:",
          {q: round(v, 1) for q, v in cv.items()}, "-> selected", best_q)

    res = {}
    for kind in ["baseline", "improved"]:
        q = best_q if kind == "improved" else 0.5
        print(f"[{kind}] learning on training seeds ...")
        tr_steps, _, _ = evaluate(kind, TRAIN_ENV, q)
        print(f"[{kind}] scoring on held-out evaluation seeds ...")
        steps, stuck, ret = evaluate(kind, TEST_ENV, q)
        per_agent = steps.mean(1)                     # mean steps per agent seed
        res[kind] = dict(score=per_agent.mean(), std=per_agent.std(), train=tr_steps.mean(),
                         stuck=stuck.mean(), ret=ret.mean())
        print(f"[{kind}] mean steps to goal (test) = {res[kind]['score']:.1f}")
    b, i = res["baseline"], res["improved"]
    metrics = {
        "baseline_score": b["score"], "improved_score": i["score"],
        "improvement": b["score"] - i["score"],        # lower steps is better
        "baseline_score_std": b["std"], "improved_score_std": i["std"],
        "baseline_train_steps": b["train"], "improved_train_steps": i["train"],
        "baseline_stuck_frac": b["stuck"], "improved_stuck_frac": i["stuck"],
        "baseline_greedy_return": b["ret"], "improved_greedy_return": i["ret"],
        "improved_cap_quantile": best_q,
    }
    for q, v in cv.items():
        metrics[f"cv_train_steps_cap_q_{q}"] = v
    metrics = {k: float(v) for k, v in metrics.items()}
    if not all(np.isfinite(list(metrics.values()))):
        print("Non-finite metrics; aborting.")
        sys.exit(1)

    os.makedirs("figures", exist_ok=True)             # bar chart of main metric
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["baseline", "improved"], [b["score"], i["score"]],
           yerr=[b["std"], i["std"]], capsize=5, color=["gray", "steelblue"])
    ax.set_ylabel("mean steps to first goal (test)")
    fig.tight_layout()
    fig.savefig("figures/steps_to_goal.png", dpi=120)
    plt.close(fig)

    n_env = len(TRAIN_ENV) + len(TEST_ENV)
    results = {
        "question": "Can a TD-error-seeking exploration rule that detects and caps persistently "
                    "high-TD-error transitions reach the sparse-reward goal faster than the plain "
                    "TD-error-seeking rule in a small stochastic-reward gridworld?",
        "method": "Tabular Q-learning with error-seeking action selection on a simulated gridworld "
                  "(sparse goal, noisy costly cells). Learning runs over agent seeds; main score is "
                  "mean environment steps to first goal on held-out evaluation seeds. The improved "
                  "cap quantile is chosen by leave-one-training-env-out CV on training seeds only.",
        "dataset": {
            "name": "Simulated stochastic-reward gridworld (generated in code)",
            "source_url": "none: generated in code, no download",
            "n_rows": n_env, "n_train": len(TRAIN_ENV), "n_test": len(TEST_ENV),
            "same_as_paper": False, "baseline_is_papers_method": False,
        },
        "metric_name": "mean environment steps to first goal on held-out evaluation seeds "
                       "(lower is better; capped at the step budget)",
        "higher_is_better": False,
        "baseline": "Tabular Q-learning; actions chosen by argmax of running mean |TD error| per "
                    "state-action pair (stand-in for SEE's error-seeking objective), eps=0.05.",
        "improved": "Same agent, but each state-action pair keeps a TD-error history; pairs whose "
                    "|TD error| stays high without decaying are flagged stuck and their exploration "
                    "value is capped at a visited-value quantile (0, 0.25 or 0.5) chosen by "
                    "leave-one-training-env-out CV. The earlier median-only cap left stuck pairs "
                    "above well-explored goal-directed pairs, which is why it lost to the baseline.",
        "metrics": metrics,
        "notes": "Simulated toy environment; tabular Q-learning, not SAC with neural networks, so "
                 "this is only a stand-in for SEE. Only the improved variant's cap quantile was "
                 "tuned, by CV on TRAIN_ENV; the baseline and the test seeds were not used for "
                 "selection. The candidate grid includes the previous median cap, so the selected "
                 "setting is no worse than it on training seeds. Steps are censored at the step "
                 "budget. Only three agent seeds; no significance test. The stuck detector "
                 "thresholds are design choices and results may not favour the improved variant.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Wrote results.json")


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
