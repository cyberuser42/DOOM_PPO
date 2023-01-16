from stable_baselines3 import PPO
from TrainAndLoggingCallback import TrainAndLoggingCallback
from DeathmatchEnv import DoomWithBots

from stable_baselines3.common import vec_env
from stable_baselines3.common.callbacks import EvalCallback

CHECKPOINT_DIR = 'train_deathmatch/'
LOG_DIR = 'log_deathmatch/'
CHECKPOINT_DIR_EVAL = 'train_deathmatch_EVAL/'
LOG_DIR_EVAL = 'log_deathmatch_EVAL/'


callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR)


def vec_env_with_bots(n_envs=2):
    return vec_env.VecTransposeImage(vec_env.DummyVecEnv([lambda: DoomWithBots(render=False)] * n_envs))

# Create environments with bots.
env = vec_env_with_bots(1)
eval_env = vec_env_with_bots(1)

eval_callback = EvalCallback(
    eval_env, 
    n_eval_episodes=5, 
    eval_freq=16384, 
    log_path=LOG_DIR_EVAL,
    best_model_save_path=CHECKPOINT_DIR_EVAL)

model = PPO('CnnPolicy', env, n_epochs=3, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=4096)

model.learn(total_timesteps=5000000, callback=[eval_callback, callback])
