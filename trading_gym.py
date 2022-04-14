#api for recup finance data
from yahoo_fin.stock_info import get_data
import gym
import gym_anytrading

# Stable baselines - rl stuff
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C,PPO,DQN

# Processing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def recuperation_donnee(name="amzn",start="12/04/2018",end="12/04/2019",interval="1wk"):
    df = get_data(name, start_date=start, end_date=end, index_as_date = True, interval=interval)
    df.columns = ["Open","High","Low","Close","Volume","5","6"]
    df.drop("5",inplace=True,axis=1)
    df.drop("6",inplace=True,axis=1)
    df.index.names = ['Date']
    return df


def creation_env(df,frame_bound=(15,200),window_size=15):
    env_maker = lambda: gym.make('stocks-v0', df=df, frame_bound=frame_bound, window_size=window_size)
    env = DummyVecEnv([env_maker])
    return env
def entrainement(env,algo=A2C,timesteps=50000):
    model = algo("MlpPolicy",env, verbose=1)
    model.learn(total_timesteps=timesteps)
    return model

def evaluation(df,model,frame_bound=(200,300),window_size=15):
    env = gym.make('stocks-v0', df=df, frame_bound=frame_bound, window_size=window_size)
    obs = env.reset()
    while True: 
        obs = obs[np.newaxis, ...]
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            #print("info", info)
            break
    return env, info
def graphique(env):
    plt.figure(figsize=(15,6))
    plt.cla()
    env.render_all()
    plt.show()

