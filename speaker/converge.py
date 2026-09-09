"""ed3 unified convergence rule: the five training scripts (finetune/train +
baselines/train_dense/mod/rt/mdf) share the same stopping criterion and constants; whoever
converges first stops first, and runs that fail to converge hit the cap.

Rule: score on a held-out subset every eval_every steps; stop when best fails to improve for
patience consecutive evals and step >= min_steps; max_steps is the hard cap (--steps means
exactly this, baselines have no annealing; for ours the anneal horizon also uses --steps,
see finetune/train.py).
"""


class StopOnPlateau:
    def __init__(self, eval_every=200, patience=3, min_steps=500, max_steps=3000):
        self.eval_every = eval_every
        self.patience = patience
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.best = None
        self.bad = 0

    def check(self, step: int, heldout_loss: float) -> bool:
        """The caller feeds held-out loss at the eval cadence; returns True = stop."""
        if self.best is None or heldout_loss < self.best - 1e-9:
            self.best = heldout_loss
            self.bad = 0
            return False
        self.bad += 1
        return step >= self.min_steps and self.bad >= self.patience

    def capped(self, step: int) -> bool:
        return step >= self.max_steps
