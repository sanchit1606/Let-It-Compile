"""
Tests for Gymnasium environment and RL setup.

Tests:
- Gym environment can be created
- Environment resets without errors
- Actions can be sampled
- Step function works
"""
import pytest


def _skip_if_no_cuda():
    try:
        from numba import cuda

        if not cuda.is_available():
            pytest.skip("CUDA not available")
    except Exception:
        pytest.skip("Numba CUDA not available")


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


def test_env_reset_step_smoke():
    """Phase 3 smoke test: env reset + one step.

    This is skip-safe:
    - skips if CUDA isn't available
    - does NOT require ncu (CUPTI) or NVML
    """

    _skip_if_no_cuda()

    from environment.kernel_env import KernelOptimizationEnv, EpisodeConfig

    cfg = EpisodeConfig(
        kernel_name="gemm",
        matrix_size=64,
        max_steps=2,
        warmup=1,
        repeats=2,
        use_cupti=False,
        use_nvml=False,
    )
    env = KernelOptimizationEnv(cfg)
    obs, info = env.reset(seed=0)

    assert env.observation_space.contains(obs)
    assert "baseline_ms" in info

    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info2 = env.step(action)

    assert env.observation_space.contains(obs2)
    assert isinstance(reward, float)
    assert terminated is False
    assert isinstance(truncated, bool)
    assert "time_ms" in info2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
