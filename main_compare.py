import trading_gym as tg
from stable_baselines3 import DQN, A2C, PPO

amzn_data = tg.recuperation_donnee(start="01/01/2017",end="01/04/2022",interval="1d")
tesla_data = tg.recuperation_donnee(name="tsla",start="01/01/2017",end="01/04/2022",interval="1d")
google_data = tg.recuperation_donnee(name="goog",start="01/01/2017",end="01/04/2022",interval="1d")
print("Nb donnée : ",len(amzn_data), amzn_data.head())
env_amz = tg.creation_env(df=amzn_data, frame_bound=(15,1000))
model_A2C = tg.entrainement(env_amz, algo=A2C, timesteps = 1_000_000)
model_PPO = tg.entrainement(env_amz, algo=PPO, timesteps = 1_000_000)
model_DQN = tg.entrainement(env_amz, algo=DQN, timesteps = 1_000_000)



env_A2C, info_A2C = tg.evaluation(df=amzn_data,model=model_A2C,frame_bound =(1000,1200))
env_PPO, info_PPO = tg.evaluation(df=amzn_data,model=model_PPO, frame_bound=(1000,1200))
env_DQN, info_DQN = tg.evaluation(df=amzn_data,model=model_DQN, frame_bound=(1000,1200))
tg.graphique(env_A2C, 'A2C-amazon')
tg.graphique(env_PPO, 'PPO-amazon')
tg.graphique(env_DQN, 'DQN-amazon')

########## Test PPO sur tesla ###################
env_PPO, info_PPO = tg.evaluation(df=tesla_data,model=model_PPO, frame_bound=(1000,1200))
tg.graphique(env_PPO, 'PPO-tesla')

########## Test PPO sur google ###################
env_PPO, info_PPO = tg.evaluation(df=google_data,model=model_PPO, frame_bound=(1000,1200))
tg.graphique(env_PPO, 'PPO-google')

