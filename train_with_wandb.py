"""Custom training script with wandb logging using RSL RL PPO."""

import argparse
import os
from datetime import datetime

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Train an RL agent with RSL RL PPO and wandb logging."
)
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=2000,
    help="Interval between video recordings (in steps).",
)
parser.add_argument(
    "--video_interval_iters",
    type=int,
    default=10,
    help="Upload a video to wandb every N training iterations.",
)
parser.add_argument(
    "--num_envs", type=int, default=4096, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--seed", type=int, default=42, help="Seed used for the environment"
)
parser.add_argument(
    "--total_timesteps", type=int, default=100000, help="Total training timesteps."
)
parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate.")
parser.add_argument(
    "--buffer_size", type=int, default=1000000, help="Replay buffer size."
)
parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")

# Wandb arguments
parser.add_argument(
    "--use_wandb", action="store_true", default=False, help="Enable wandb logging."
)
parser.add_argument(
    "--wandb_project", type=str, default="isaaclab-training", help="Wandb project name."
)
parser.add_argument(
    "--wandb_entity", type=str, default=None, help="Wandb entity/username."
)
parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry

# Import RSL RL
try:
    from rsl_rl.runners import OnPolicyRunner
    from rsl_rl.algorithms import PPO
    from rsl_rl.modules import ActorCritic

    RSL_RL_AVAILABLE = True
except ImportError:
    RSL_RL_AVAILABLE = False
    print("[WARNING] rsl_rl not installed. Install with: pip install rsl-rl")

# Import wandb
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WARNING] wandb not installed. Install with: pip install wandb")


class WandbLogger:
    """Simple wandb logger for training."""

    def __init__(self, project, entity=None, name=None, config=None):
        if not WANDB_AVAILABLE:
            raise ImportError("wandb not installed")

        self.run = wandb.init(
            project=project, entity=entity, name=name, config=config, reinit=True,
            sync_tensorboard=True,
        )
        print(f"[INFO] Wandb logging enabled: {self.run.url}")

    def log(self, metrics, step=None):
        """Log metrics to wandb."""
        if self.run:
            wandb.log(metrics, step=step)

    def finish(self):
        """Finish wandb run."""
        if self.run:
            wandb.finish()


def find_and_upload_latest_video(video_dir, wandb_run, step, already_uploaded):
    """Find the most recent .mp4 in video_dir and upload it to wandb if not already uploaded."""
    import glob
    mp4s = sorted(glob.glob(os.path.join(video_dir, "**", "*.mp4"), recursive=True))
    new_videos = [p for p in mp4s if p not in already_uploaded]
    if not new_videos:
        return already_uploaded
    latest = new_videos[-1]
    print(f"[INFO] Uploading video to wandb: {latest}")
    wandb.log({"rollout/video": wandb.Video(latest, fps=30, format="mp4")}, step=step)
    already_uploaded.add(latest)
    return already_uploaded


class RslRlVecEnvWrapper:
    """Wrapper to make IsaacLab environment compatible with rsl_rl."""

    def __init__(self, env):
        self._env = env
        # Copy attributes from wrapped env (these won't be forwarded via __getattr__)
        self.num_envs = env.num_envs
        self.num_actions = (
            env.action_space.shape[1]
            if hasattr(env.action_space, "shape")
            else env.action_space.n
        )
        self.max_episode_length = env.max_episode_length
        self.episode_length_buf = env.episode_length_buf
        self.device = env.device
        self.cfg = env.cfg
        self._initializing = False
        # Reset the environment to get initial observations
        obs, extras = env.reset()
        self.obs_buf = obs

    def _obs_to_tensor(self, obs):
        """Convert dict or array observations to a flat tensor."""
        if isinstance(obs, dict):
            obs_list = []
            for key in sorted(obs.keys()):
                obs_val = obs[key]
                if obs_val is None:
                    continue
                if isinstance(obs_val, torch.Tensor):
                    obs_list.append(obs_val.view(obs_val.shape[0], -1))
                else:
                    obs_tensor = torch.tensor(obs_val, device=self.device)
                    obs_list.append(obs_tensor.view(obs_tensor.shape[0], -1))
            if obs_list:
                return torch.cat(obs_list, dim=1)
            return torch.zeros(self.num_envs, 1, device=self.device)
        elif not isinstance(obs, torch.Tensor):
            return torch.tensor(obs, device=self.device)
        return obs

    def _obs_is_empty(self, obs):
        """Check if observations are empty (None, empty dict, or dict with no valid tensors)."""
        if obs is None:
            return True
        if isinstance(obs, dict) and (not obs or all(v is None for v in obs.values())):
            return True
        return False

    def get_observations(self):
        """Return observations and extras tuple (compatible with rsl_rl)."""
        # If obs_buf not yet initialized or empty, reset the environment first
        if self._obs_is_empty(self.obs_buf) and self._obs_is_empty(getattr(self._env, "obs_buf", None)) and not self._initializing:
            self._initializing = True
            self.reset(torch.arange(self.num_envs, device=self.device))
            self._initializing = False

        # Get observations from the wrapper's obs_buf (updated after each step)
        obs = self.obs_buf
        if self._obs_is_empty(obs):
            obs = getattr(self._env, "obs_buf", None)

        if self._obs_is_empty(obs):
            raise RuntimeError("Cannot obtain observations from environment - obs_buf is empty")

        obs = self._obs_to_tensor(obs)

        # Get extras from environment if available
        extras = self._env.extras if hasattr(self._env, "extras") else {}

        # Ensure extras has the 'observations' key that rsl_rl expects
        if "observations" not in extras:
            extras["observations"] = {}

        # Return as tuple: (obs, extras) - rsl_rl expects this
        return obs, extras

    def step(self, actions):
        """Step the environment."""
        # IsaacLab step returns: obs, reward, terminated, truncated, extras
        obs, reward, terminated, truncated, extras = self._env.step(actions)
        # Store raw obs in obs_buf for get_observations()
        self.obs_buf = obs
        # Convert obs to tensor for rsl_rl
        obs_tensor = self._obs_to_tensor(obs)
        # rsl_rl expects: obs, reward, done, extras
        done = terminated | truncated
        return obs_tensor, reward, done, extras

    def reset(self, env_ids=None):
        """Reset the environment."""
        obs, extras = self._env.reset()
        self.obs_buf = obs
        return self.get_observations()

    def __getattr__(self, name):
        """Forward attribute access to the wrapped environment."""
        # Don't forward these attributes - they're defined on the wrapper itself
        if name in ["_env"]:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        return getattr(self._env, name)


def main():
    """Train with RSL RL PPO agent and wandb logging."""

    if not RSL_RL_AVAILABLE:
        print("[ERROR] rsl_rl is required for PPO training")
        return

    # Parse task configuration manually
    if args_cli.task is None:
        print("[ERROR] No task specified. Use --task=<task_name>")
        return

    print(f"[INFO] Loading task: {args_cli.task}")

    # Parse task config using IsaacLab's utility
    try:
        # Load environment config from registry
        env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
        # Load agent config from registry
        agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    except Exception as e:
        print(f"[ERROR] Failed to parse task config: {e}")
        print("[INFO] Trying alternative method...")
        # Fallback: try to import directly
        try:
            from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.flat_env_cfg import (
                AnymalDFlatEnvCfg,
            )
            from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.agents.rsl_rl_ppo_cfg import (
                AnymalDFlatPPORunnerCfg,
            )

            env_cfg = AnymalDFlatEnvCfg()
            agent_cfg = AnymalDFlatPPORunnerCfg()
        except Exception as e2:
            print(f"[ERROR] Fallback also failed: {e2}")
            return

    # Set the environment seed
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )
    env_cfg.scene.num_envs = args_cli.num_envs

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl_ppo", args_cli.task.replace("-", "_"))
    log_root_path = os.path.abspath(log_root_path)
    os.makedirs(log_root_path, exist_ok=True)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # specify directory for logging runs
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args_cli.wandb_run_name:
        log_dir = f"{args_cli.wandb_run_name}_{log_dir}"
    log_dir = os.path.join(log_root_path, log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Initialize wandb if enabled
    wandb_logger = None
    if args_cli.use_wandb and WANDB_AVAILABLE:
        run_name = (
            args_cli.wandb_run_name
            or f"{args_cli.task}_RSLRL_PPO_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        wandb_config = {
            "task": args_cli.task,
            "num_envs": args_cli.num_envs,
            "seed": args_cli.seed,
            "total_timesteps": args_cli.total_timesteps,
            "learning_rate": args_cli.learning_rate,
            "algorithm": "PPO",
            "library": "RSL_RL",
        }
        wandb_logger = WandbLogger(
            project=args_cli.wandb_project,
            entity=args_cli.wandb_entity,
            name=run_name,
            config=wandb_config,
        )

    # Create the environment
    print("[INFO] Creating environment...")
    try:
        from isaaclab.envs import ManagerBasedRLEnv

        env = ManagerBasedRLEnv(cfg=env_cfg)
        print(f"[INFO] Created raw environment: {type(env).__name__}")
        # Wrap the environment for rsl_rl compatibility
        env = RslRlVecEnvWrapper(env)
        print(f"[INFO] Wrapped environment: {type(env).__name__}")
        # Verify wrapper has get_observations method
        if hasattr(env, "get_observations"):
            print("[INFO] Wrapper has get_observations method: YES")
        else:
            print("[ERROR] Wrapper missing get_observations method!")
    except Exception as e:
        print(f"[ERROR] Failed to create environment: {e}")
        import traceback

        traceback.print_exc()
        return

    # Use the agent configuration from the registry
    # Convert to dict if it's a config object
    if hasattr(agent_cfg, "to_dict"):
        runner_cfg = agent_cfg.to_dict()
    else:
        # Assume it's already a dict
        runner_cfg = agent_cfg

    # Update with command-line overrides if provided
    if hasattr(runner_cfg, "__dict__"):
        runner_cfg = vars(runner_cfg)

    # Override max_iterations if total_timesteps was specified
    if args_cli.total_timesteps:
        runner_cfg["max_iterations"] = args_cli.total_timesteps // (args_cli.num_envs * runner_cfg.get("num_steps_per_env", 24))

    # Override learning rate if specified
    if hasattr(runner_cfg.get("algorithm", {}), "learning_rate"):
        runner_cfg["algorithm"]["learning_rate"] = args_cli.learning_rate
    elif isinstance(runner_cfg.get("algorithm"), dict):
        runner_cfg["algorithm"]["learning_rate"] = args_cli.learning_rate

    # Verify environment is wrapped correctly
    print(f"[INFO] Environment type before runner: {type(env).__name__}")
    if hasattr(env, "get_observations"):
        print("[INFO] Environment has get_observations: YES")
    else:
        print("[ERROR] Environment missing get_observations!")
        return

    # Create RSL RL runner
    runner = OnPolicyRunner(env, runner_cfg, log_dir, device=env_cfg.sim.device)

    # Train the agent
    num_iterations = runner_cfg["max_iterations"]
    video_interval = args_cli.video_interval_iters if args_cli.video else 0
    print(
        f"[INFO] Starting RSL RL PPO training for {args_cli.total_timesteps} timesteps ({num_iterations} iterations)..."
    )

    if not args_cli.video or not wandb_logger:
        # No video recording — run all at once
        runner.learn(num_iterations)
    else:
        # Chunked training loop with periodic video uploads
        uploaded_videos = set()
        iters_done = 0
        while iters_done < num_iterations:
            chunk = min(video_interval, num_iterations - iters_done)
            runner.learn(chunk)
            iters_done += chunk
            uploaded_videos = find_and_upload_latest_video(
                log_dir, wandb_logger.run, step=iters_done, already_uploaded=uploaded_videos
            )
            print(f"[INFO] Completed {iters_done}/{num_iterations} iterations")

    # Log final metrics to wandb
    if wandb_logger:
        wandb_logger.log({"train/completed": True})
        wandb_logger.finish()

    print("[INFO] Training completed!")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
