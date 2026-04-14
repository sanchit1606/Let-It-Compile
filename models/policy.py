"""Custom policy network for PPO training on kernel optimization.

This is optional; SB3's default MlpPolicy can also be used.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN, create_mlp


class KernelOptimizationFeatureExtractor(BaseFeaturesExtractor):
    """Feature extractor for bounded [0,1] observation vectors.

    Input: flattened observation (CUPTI + NVML + kernel one-hot + prev action)
    Output: feature vector for actor/critic heads
    """

    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input = int(np.prod(observation_space.shape))
        self.net = nn.Sequential(
            nn.Linear(n_input, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class KernelOptimizationPolicy(ActorCriticPolicy):
    """Custom actor-critic policy for kernel optimization.

    Uses a simple shared feature extractor + separate actor/critic heads.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule: Callable[[int], float],
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        *args,
        **kwargs,
    ):
        if net_arch is None:
            net_arch = [dict(pi=[64, 64], vf=[64, 64])]

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        """Build shared feature extractor + actor/critic heads."""
        self.mlp_extractor = KernelOptimizationFeatureExtractor(
            self.observation_space,
            features_dim=128,
        )

        # Actor and critic heads
        self.policy_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.value_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
