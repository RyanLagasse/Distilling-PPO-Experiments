# distill_lunar_lander.py
import os
import gymnasium as gym
import imageio
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
from stable_baselines3 import PPO
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1) Save your teacher (run once, or reuse your existing model file)
def save_teacher(env_name="LunarLander-v3", timesteps=500_000, teacher_path="ppo_teacher"):
    env = gym.make(env_name)
    teacher = PPO("MlpPolicy", env, verbose=1)
    teacher.learn(total_timesteps=timesteps)
    teacher.save(teacher_path)
    env.close()

# 2) Wrapper that replaces reward with similarity-to-teacher
class MimicRewardWrapper(gym.Wrapper):
    def __init__(self, env, teacher_model):
        super().__init__(env)
        self.teacher = teacher_model

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        # Ask teacher for its action (deterministic)
        teacher_action, _ = self.teacher.predict(obs, deterministic=False)
        # Define mimic reward: +1 for exact match, else 0
        mimic_reward = 1.0 if action == teacher_action else 0.0
        return obs, mimic_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# 3) Train a smaller “student” PPO
def train_student(teacher_path="ppo_teacher",
                  student_path="ppo_student",
                  arch_params=[64, 64],
                  env_name="LunarLander-v3",
                  timesteps=50_000):

    teacher = PPO.load(teacher_path)
    base_env = gym.make(env_name)
    # Wrap with mimic reward
    env = MimicRewardWrapper(base_env, teacher)
    # Make networks tiny: 2 hidden layer of the archetecture we make up (designed for 2 layer but can take a differnt number)
    policy_kwargs = dict(net_arch=[dict(pi=arch_params, vf=arch_params)])
    student = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    student.learn(total_timesteps=timesteps)
    student.save(student_path)
    env.close()

# 4) Record test runs and save as GIF
def make_gif(gif_path="test.gif",
             model_path="ppo_student",
             env_name="LunarLander-v3",
             max_steps=1000,
             fps=60):
    # Video folder is temporary; RecordVideo will save mp4s there
    vid_folder = "videos"
    os.makedirs(vid_folder, exist_ok=True)
    # Render as rgb_array so we can capture frames
    env = RecordVideo(
        gym.make(env_name, render_mode="rgb_array"),
        video_folder=vid_folder,
        name_prefix="student")
    model = PPO.load(model_path)

    frames = []
    obs, _ = env.reset()
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        frame = env.render()
        frames.append(frame)
        if done:
            break
    env.close()

    # Combine frames into a GIF
    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"Saved GIF to {gif_path}")

def test_agent(model_path="ppo_lunar_lander", env_name="LunarLander-v3", episodes=5):
    """
    Load a trained PPO agent and run test episodes with rendering.
    """
    env = gym.make(env_name, render_mode="human")
    model = PPO.load(model_path)
    
    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            total_reward += reward
            done = terminated or truncated
        
        print(f"Test Episode {ep} Reward: {total_reward:.2f}")

if __name__ == "__main__":
    # Train the teacher model
    # save_teacher()

    # Distill a student from teacher model
    # train_student(teacher_path="ppo_teacher_v2",
    #               student_path="ppo_student_v3",
    #               env_name="LunarLander-v3",
    #               timesteps=50_000)

    # Record a test run of the distilled student
    # make_gif("version-6.gif", model_path="ppo_student_v3")
    # make_gif("version-7.gif", model_path="ppo_teacher_v2")


    # test_agent(model_path="ppo_student_v2",episodes=5)


    # experiments
    param_experiments = [
        # {"name": "32_64", "net_arch": [dict(pi=[64,32], vf=[64,32])]},
        # {"name": "256_256", "arch_params": [256,256]},
        # {"name": "128_128", "arch_params": [128,128]},
        # {"name": "64_64", "arch_params": [64,64]},
        # {"name": "64_32", "arch_params": [64,32]},
        # {"name": "64_16", "arch_params": [64,16]},
        # {"name": "64_8", "arch_params": [64,8]},
        {"name": "64_4", "arch_params": [64,4]},
        {"name": "64_2", "arch_params": [64,2]},
        {"name": "64_1", "arch_params": [64,1]},
    ]

    for params in param_experiments:
        train_student(
            teacher_path="ppo_teacher_v2",
            student_path=f"ppo_student_v3_{params['name']}",
            arch_params=params["arch_params"],
        )
        make_gif(
            gif_path=f"params_{params['name']}.gif",
            model_path=f"ppo_student_v3_{params['name']}",
        )
        test_agent(model_path=f"ppo_student_v3_{params['name']}",episodes=5)