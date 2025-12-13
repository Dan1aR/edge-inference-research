from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class ScheduleStage:
    group: str
    milestone: float  # fraction of total steps


class ProgressiveSchedule:
    """Progressive enablement schedule for custom layers.

    The schedule string uses ``group@fraction`` entries separated by commas, e.g.::

        attn@0.2,mlp@0.4,qkv@0.6,head@0.8
    """

    def __init__(self, stages: List[ScheduleStage]):
        self.stages = sorted(stages, key=lambda s: s.milestone)

    @classmethod
    def from_string(cls, value: str) -> "ProgressiveSchedule":
        stages: List[ScheduleStage] = []
        if not value:
            return cls(stages)
        for item in value.split(","):
            group, frac = item.split("@")
            stages.append(ScheduleStage(group=group.strip(), milestone=float(frac)))
        return cls(stages)

    def enabled_groups(self, step: int, total_steps: int) -> List[str]:
        if total_steps <= 0:
            return []
        frac = step / float(total_steps)
        return [s.group for s in self.stages if frac >= s.milestone]

    def enabled_blocks(self, step: int, total_steps: int, *, blocks_per_stage: int, num_blocks_total: int) -> int:
        if blocks_per_stage <= 0:
            return num_blocks_total
        stages_completed = len(self.enabled_groups(step, total_steps))
        return min(num_blocks_total, stages_completed * blocks_per_stage)
