import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

def evaluate_model(model_path: str,
                   env_name: str = "LunarLander-v3",
                   episodes: int = 10) -> dict:
    """
    Load a trained PPO model and run `episodes` test rollouts.
    Returns a dict with mean, std, min, max rewards.
    """
    env = gym.make(env_name)
    model = PPO.load(model_path)
    
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_r += reward
            done = terminated or truncated
        rewards.append(total_r)
    env.close()
    
    arr = np.array(rewards)
    return {
        "model": model_path,
        "mean_reward": float(arr.mean()),
        "std_reward": float(arr.std()),
        "min_reward": float(arr.min()),
        "max_reward": float(arr.max()),
    }

def evaluate_models(model_paths: list[str],
                    env_name: str = "LunarLander-v3",
                    episodes: int = 10) -> pd.DataFrame:
    """
    Evaluate each model in `model_paths` over `episodes` runs
    and return a DataFrame summarizing their performance.
    """
    records = []
    for path in model_paths:
        stats = evaluate_model(path, env_name=env_name, episodes=episodes)
        records.append(stats)
    return pd.DataFrame(records).set_index("model")

# Example usage:
if __name__ == "__main__":
    my_models = [
        "ppo_teacher_v2.zip",
        "ppo_student_v3_256_256.zip",
        "ppo_student_v3_128_128.zip",
        "ppo_student_v3_64_64.zip",
        "ppo_student_v3_64_32.zip",
        "ppo_student_v3_64_16.zip",
        "ppo_student_v3_64_8.zip",
        "ppo_student_v3_64_4.zip",
        "ppo_student_v3_64_2.zip",
        "ppo_student_v3_64_1.zip",
        # ... whatever else you’ve generated
    ]
    df = evaluate_models(my_models, episodes=50)
    print(df)
    # Optionally save to CSV:
    df.to_csv("model_comparison.csv")
