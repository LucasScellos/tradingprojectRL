import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from yahoo_fin.stock_info import get_data
import gym
import gym_anytrading
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, PPO, DQN


def recuperation_donnee(
    name="amzn", start="12/04/2018", end="12/04/2019", interval="1wk"
):
    """
    Retrieve financial data and process it.

    Parameters
    ----------
    name : str, optional
        The stock symbol, by default "amzn".
    start : str, optional
        Start date for data retrieval, by default "12/04/2018".
    end : str, optional
        End date for data retrieval, by default "12/04/2019".
    interval : str, optional
        Interval for data retrieval, by default "1wk".

    Returns
    -------
    pandas.DataFrame
        Processed financial data.
    """
    df = get_data(
        name, start_date=start, end_date=end, index_as_date=True, interval=interval
    )
    df.columns = ["Open", "High", "Low", "Close", "Volume", "5", "6"]
    df.drop("5", inplace=True, axis=1)
    df.drop("6", inplace=True, axis=1)
    df.index.names = ["Date"]
    return df


def creation_env(df, frame_bound=(15, 200), window_size=15):
    """
    Create a stock trading environment.

    Parameters
    ----------
    df : pandas.DataFrame
        Financial data.
    frame_bound : tuple, optional
        Range of the data to be used, by default (15, 200).
    window_size : int, optional
        Number of past observations to consider, by default 15.

    Returns
    -------
    stable_baselines3.common.vec_env.DummyVecEnv
        Created stock trading environment.
    """
    env_maker = lambda: gym.make(
        "stocks-v0", df=df, frame_bound=frame_bound, window_size=window_size
    )
    env = DummyVecEnv([env_maker])
    return env


def entrainement(env, algo=A2C, timesteps=50000):
    """
    Train an RL model on the stock trading environment.

    Parameters
    ----------
    env : stable_baselines3.common.vec_env.DummyVecEnv
        Stock trading environment.
    algo : type, optional
        RL algorithm to be used, by default A2C.
    timesteps : int, optional
        Number of timesteps for training, by default 50000.

    Returns
    -------
    stable_baselines3.a2c.a2c.A2C or stable_baselines3.ppo.ppo.PPO or stable_baselines3.dqn.dqn.DQN
        Trained RL model.
    """
    model = algo("MlpPolicy", env, verbose=1)
    model


def evaluation(df, model, frame_bound=(200, 300), window_size=15):
    """
    Evaluate the trained model on a specified stock data.

    Parameters
    ----------
    df : pd.DataFrame
        The stock data.
    model : StableBaselines3 model
        The trained model.
    frame_bound : tuple, optional
        The starting and ending points of the data, by default (200, 300)
    window_size : int, optional
        The size of the window for the stock environment, by default 15

    Returns
    -------
    env : gym environment
        The environment used for the evaluation.
    info : dict
        The information of the last step of the evaluation.
    """
    env = gym.make("stocks-v0", df=df, frame_bound=frame_bound, window_size=window_size)
    obs = env.reset()
    while True:
        obs = np.expand_dims(obs, axis=0)
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            break
    return env, info


def graphique(env, title="A2C"):
    """
    Plot the evaluation results.

    Parameters
    ----------
    env : gym environment
        The environment used for the evaluation.
    title : str, optional
        The title of the plot, by default "A2C"

    Returns
    -------
    None
    """
    plt.figure(figsize=(15, 6))
    plt.cla()
    env.render_all()
    plt.suptitle(title)
    plt.show()
