import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

class StockTradingEnv(gym.Env):
    """
    A stock trading environment for OpenAI gym
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        
        self.df = df
        self.MAX_STEPS = len(df) - 1
        
        # Actions of the format Buy x%, Sell x%, Hold, etc.
        # For simplicity: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [Open, High, Low, Close, Volume, RSI, MACD, ...]
        # Assuming normalized data is passed or we normalize inside
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(df.columns),), dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.net_worth = 10000
        self.max_net_worth = 10000
        
        return self._next_observation(), {}

    def _next_observation(self):
        obs = self.df.iloc[self.current_step].values
        return obs.astype(np.float32)

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        
        self.current_step += 1
        
        if self.current_step > len(self.df) - 2:
            self.current_step = 0 # Loop or end
            terminated = True
        else:
            terminated = False
            
        truncated = False
        
        reward = self.net_worth - 10000 # Simple reward: profit
        # Or reward = self.net_worth - self.prev_net_worth
        
        obs = self._next_observation()
        
        return obs, reward, terminated, truncated, {}

    def _take_action(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        
        # 0: Hold, 1: Buy, 2: Sell
        if action == 1: # Buy
            # Buy with 10% of balance
            amount_to_invest = self.balance * 0.1
            if amount_to_invest > current_price:
                shares = amount_to_invest / current_price
                self.balance -= amount_to_invest
                self.shares_held += shares
                
        elif action == 2: # Sell
            # Sell 10% of shares
            shares_to_sell = self.shares_held * 0.1
            if shares_to_sell > 0:
                self.balance += shares_to_sell * current_price
                self.shares_held -= shares_to_sell
                
        self.net_worth = self.balance + self.shares_held * current_price

def train_rl_agent(env, timesteps=10000):
    """
    Trains a PPO agent.
    """
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=timesteps)
    return model

def evaluate_agent(model, env):
    """
    Evaluates the trained agent and returns the history.
    """
    obs, _ = env.reset()
    history = {
        'net_worth': [],
        'actions': [],
        'price': []
    }
    
    terminated = False
    truncated = False
    
    while not terminated and not truncated:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        history['net_worth'].append(env.net_worth)
        history['actions'].append(action)
        history['price'].append(env.df.iloc[env.current_step]['Close'])
        
    return history
