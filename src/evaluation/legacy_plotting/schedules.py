"""
Utility callbacks for Hopper sim-to-real experiments
====================================================

This module defines two light-weight callbacks that can be attached to
*any* Stable-Baselines3 (SB3) on-policy learner:

1) EntropyScheduler
   • Linearly decays the entropy coefficient `ent_coef`
     from `start_coef` → `end_coef` during training.
   • Works out-of-the-box with PPO, A2C, TRPO in SB3.

2) CurriculumMassRandomizer
   • Gradually widens the uniform domain-randomization
     half-width on link masses (Curriculum Domain Randomization, CDR).
   • Assumes each parallel environment exposes a
     `set_mass_width(width: float)` method.

Both classes inherit from `BaseCallback`, so they slot directly into
`model.learn()` via the `callback=[…]` argument.

Author: <your-name> · GPL-3.0-or-later · 2025-06-22
"""
from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


# --------------------------------------------------------------------------- #
#                               Entropy Scheduler                              #
# --------------------------------------------------------------------------- #
class EntropyScheduler(BaseCallback):
    """
    Linearly interpolates `ent_coef` from `start_coef` to `end_coef`
    over `total_timesteps`.

    Example
    -------
    >>> sched = EntropyScheduler(0.01, 1e-4, 5_000_000)
    >>> model.learn(total_timesteps=5_000_000, callback=[sched])
    """

    def __init__(
        self,
        start_coef: float,
        end_coef: float,
        total_timesteps: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.start_coef = float(start_coef)
        self.end_coef = float(end_coef)
        self.total_steps = int(total_timesteps)

    # Called by SB3 every environment step
    def _on_step(self) -> bool:
        # Fraction of training completed ∈ [0, 1]
        frac = min(1.0, self.num_timesteps / float(self.total_steps))
        new_coef = (1.0 - frac) * self.start_coef + frac * self.end_coef

        # PPO / A2C keep ent-coef in a tensor; slice assignment is fastest
        # (works even with multiple policies during eval callback)
        try:
            self.model.ent_coef_tensor[:] = new_coef
        except AttributeError:        # fallback for custom agents
            if hasattr(self.model, "ent_coef"):
                self.model.ent_coef = new_coef
        return True  # keep training


# --------------------------------------------------------------------------- #
#                     Curriculum-based Mass Randomization                     #
# --------------------------------------------------------------------------- #
class CurriculumMassRandomizer(BaseCallback):
    """
    Expands the mass-randomization range from 0 to `max_width`
    during training. Requires each env to implement

        env.unwrapped.set_mass_width(width: float)

    where *width* is the half-width of the uniform distribution, e.g.
    width = 0.25 means masses are sampled from
    [0.75 · m_nominal, 1.25 · m_nominal].

    Example
    -------
    >>> cdr = CurriculumMassRandomizer(max_width=0.40,
    ...                                total_timesteps=5_000_000)
    >>> model.learn(total_timesteps=5_000_000,
    ...             callback=[cdr, EntropyScheduler(...)])

    Tip: If you log `width` inside the env, you get a ready-made
    curriculum plot for the appendix.
    """

    def __init__(
        self,
        max_width: float = 0.40,
        total_timesteps: int = 5_000_000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.max_width = float(max_width)
        self.total_steps = int(total_timesteps)

    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / float(self.total_steps))
        width = frac * self.max_width

        # Update every parallel env in the VecEnv
        for env in self.training_env.envs:
            # unwrap() to bypass TimeLimit, etc.
            try:
                env.unwrapped.set_mass_width(width)
            except AttributeError:
                # silently ignore envs that do not define the hook
                pass
        return True
