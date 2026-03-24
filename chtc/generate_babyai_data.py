import gymnasium as gym
import minigrid  # Required to register BabyAI environments
import pandas as pd
import os

# 1. Define where you want to save the data
SAVE_DIR = os.path.expanduser("~/data/babyai")
os.makedirs(SAVE_DIR, exist_ok=True)

def generate_dataset(env_name, num_samples, output_filename):
    print(f"Generating {num_samples} samples for {env_name}...")
    
    # Initialize the BabyAI environment
    env = gym.make(env_name)
    data = []
    
    for i in range(num_samples):
        # Reset the environment to get a brand new procedural layout
        obs, info = env.reset(seed=i)
        
        # In BabyAI, the 'mission' is the text instruction (e.g., "go to the red ball")
        mission = obs['mission']
        
        # Format the prompt exactly how your LLM expects to see it on turn 1
        prompt = f"You are an agent in a grid world. Your mission is: {mission}. What is your first action?"
        
        # verl expects the data in a dictionary, usually looking for a 'prompt' key
        data.append({
            "prompt": prompt,
            "id": f"babyai_goto_{i}"
        })
        
    # Convert to a Pandas DataFrame and save as a Parquet file
    df = pd.DataFrame(data)
    save_path = os.path.join(SAVE_DIR, output_filename)
    df.to_parquet(save_path)
    print(f"Successfully saved to {save_path}\n")

if __name__ == "__main__":
    # The standard 'GoTo' local environment in BabyAI
    ENV_ID = "BabyAI-GoToLocal-v0" 
    
    # Generate 10,000 unique starting states for training
    generate_dataset(ENV_ID, 10000, "train.parquet")
    
    # Generate 500 unique starting states for testing/validation
    generate_dataset(ENV_ID, 500, "test.parquet")