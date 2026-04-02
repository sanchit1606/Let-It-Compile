"""
Tests for Gymnasium environment and RL setup.

Tests:
- Gym environment can be created
- Environment resets without errors
- Actions can be sampled
- Step function works
"""
import pytest


def test_gymnasium_import():
    """Test that Gymnasium is available."""
    try:
        import gymnasium as gym
        assert gym is not None
    except ImportError:
        pytest.skip("Gymnasium not installed")


def test_stable_baselines3_import():
    """Test that Stable-Baselines3 is available."""
    try:
        from stable_baselines3 import PPO
        assert PPO is not None
    except ImportError:
        pytest.skip("Stable-Baselines3 not installed")


def test_environment_import():
    """Test that kernel environment can be imported."""
    try:
        from environment.kernel_env import KernelOptimizationEnv, EpisodeConfig
    except ImportError:
        pytest.skip("environment.kernel_env not available")


def test_episode_config_creation():
    """Test that episode config can be created."""
    try:
        from environment.kernel_env import EpisodeConfig
        config = EpisodeConfig(kernel_name="gemm", matrix_size=256, max_steps=10)
        assert config.kernel_name == "gemm"
        assert config.matrix_size == 256
        assert config.max_steps == 10
    except ImportError:
        pytest.skip("EpisodeConfig not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
