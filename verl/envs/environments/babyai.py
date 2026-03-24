# In verl/envs/babyai.py (you'll need to create this wrapper)

class TimeAwareBabyAI:
    def __init__(self, task="BabyAI-MixedTrainLocal-v0/goto"):
        self.env = gym.make(task)
        self.time_budget = 100  # 100 time steps
        self.time_elapsed = 0
        
    def reset(self):
        obs = self.env.reset()
        self.time_elapsed = 0
        return obs
        
    def step(self, action, tokens_generated):
        # Time cost = number of tokens generated
        time_cost = tokens_generated / 100  # 1 step per 100 tokens
        self.time_elapsed += time_cost
        
        # Penalty for running out of time
        time_penalty = 0
        if self.time_elapsed > self.time_budget:
            time_penalty = -1.0
            done = True
            return obs, time_penalty, done, info
        
        # Execute action in environment
        obs, reward, done, info = self.env.step(action)
        
        # Adjusted reward = task reward - time penalty
        adjusted_reward = reward - (0.01 * time_cost)
        
        return obs, adjusted_reward, done, info