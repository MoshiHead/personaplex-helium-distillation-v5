# SPDX-License-Identifier: MIT
"""The P1->P4 training schedule: loss weights, data source (teacher-forced vs.
on-policy student rollouts), depth-transformer freeze state, and learning-rate
multiplier, all as a function of training progress in [0, 1].
"""

from dataclasses import dataclass
import typing as tp


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    start: float  # inclusive, fraction of total training
    end: float  # exclusive (1.0 for the last phase)
    alpha: float  # CE weight
    beta: float  # KL weight
    gamma: float  # L_bridge weight
    delta: float  # L_hidden weight
    epsilon: float  # L_speaker weight
    on_policy: bool  # student rollouts vs. teacher-forced data
    depth_transformer_unfrozen: bool
    depth_transformer_lr_mult: float = 0.0


PHASES: tuple[PhaseSpec, ...] = (
    PhaseSpec("P1_align", 0.00, 0.25, alpha=0.0, beta=0.3, gamma=2.0, delta=1.0, epsilon=0.0,
              on_policy=False, depth_transformer_unfrozen=False),
    PhaseSpec("P2_behavior", 0.25, 0.85, alpha=0.3, beta=1.0, gamma=0.5, delta=0.3, epsilon=0.0,
              on_policy=False, depth_transformer_unfrozen=False),
    PhaseSpec("P3_on_policy", 0.85, 0.95, alpha=0.3, beta=1.0, gamma=0.2, delta=0.05, epsilon=0.2,
              on_policy=True, depth_transformer_unfrozen=False),
    PhaseSpec("P4_polish", 0.95, 1.00, alpha=0.5, beta=0.6, gamma=0.1, delta=0.0, epsilon=0.5,
              on_policy=True, depth_transformer_unfrozen=True, depth_transformer_lr_mult=0.1),
)

# Why these boundaries moved (was 0.15 / 0.60 / 0.90 / 1.00):
#   * P1 "Align" was 15% of the run -- 139 steps on the 925-step run -- and with alpha=0 it applied no
#     hard-label CE at all. The student enters P2 still far from the teacher. P1 is now 25%.
#   * P3/P4 (on-policy) were 40% of the STEPS, but an on-policy step is ~7x slower and processes ~half the
#     frames, so they consumed ~half the wall clock for ~11% of the frames (measured: 88.1 s/step and
#     188 frames/s on-policy vs 20.6 s/step and 1600 frames/s teacher-forced). On-policy rollouts fix speech
#     BEHAVIOUR -- turn-taking, not running out of things to say -- which is the one thing the student already
#     did well. They are now 15% of the steps, so the budget goes to the objective that was failing.
#   * alpha in P2 went 0.2 -> 0.3: with the text content weighting in distill/train.py the hard-label CE is
#     now mostly about words rather than padding, so it is worth more.
#   * beta now weights the AUDIO KL only; the text KL has its own weight (`--beta-text`).


def phase_at(progress: float) -> PhaseSpec:
    """`progress`: fraction of training complete, in [0, 1]."""
    assert 0.0 <= progress <= 1.0, progress
    for phase in PHASES:
        if phase.start <= progress < phase.end:
            return phase
    return PHASES[-1]  # progress == 1.0 falls into the last phase


class TrainingSchedule:
    """Stateful wrapper: given the current step and total steps, exposes the
    active phase and whether a phase transition just happened (so the caller
    can e.g. unfreeze the depth transformer and reset its optimizer state
    exactly once, at the P3->P4 boundary).
    """

    def __init__(self, total_steps: int):
        assert total_steps > 0
        self.total_steps = total_steps
        self._last_phase_name: tp.Optional[str] = None

    def progress(self, step: int) -> float:
        return min(1.0, max(0.0, step / self.total_steps))

    def step(self, step: int) -> tuple[PhaseSpec, bool]:
        """Returns (phase, phase_just_changed)."""
        phase = phase_at(self.progress(step))
        changed = phase.name != self._last_phase_name
        self._last_phase_name = phase.name
        return phase, changed

    def state_dict(self) -> dict:
        return {"total_steps": self.total_steps, "last_phase_name": self._last_phase_name}

    def load_state_dict(self, state: dict):
        self.total_steps = state["total_steps"]
        self._last_phase_name = state["last_phase_name"]
