Ce projet a été réalisé pendant mon cours de MTI880 pendant ma maitrise à l'ETS en Technologie de l'Information. 

Repris code de [Gym AnyTrading de AminHP](https://github.com/AminHP/gym-anytrading) en l'adaptant à notre cas, prédiction de stock. 

Pour présenter notre projet, nous avons réaliser un module principale nommé *trading_gym.py* qui contient les fonctions nécessaires à la démonstration de notre projet. Nous avons ensuite crée plusiuers notebook permettant de démontrer plusieurs points de notre projet : performance, vitesse d'apprentissage, généralisation des modèles.


# Commande 
Afin d'adopter une démarche modulaire, nous avons regroupé nos fonctions, on peux les utiliser comme suit : 
* **recuperation_donnee(name="amzn",start="12/04/2018",end="12/04/2019",interval="1wk")**  
  La fonction retourne un dataframe pandas avec comme index la date et comme colomne : "Open","High","Low","Close","Volume". 
> `name`: nom de l'action (nom utilisé dans le projet : *amzn, aapl, msft, goog, tsla, fb*)  
> `start`: date début au format JJ/MM/YYYY  
> `end`: date fin au format JJ/MM/YYYY  
> `interval`: interval entre chaque mesure, 1d", "1wk", "1mo", or "1m" for daily, weekly, monthly, or minute data  

* **creation_env(df,frame_bound=(15,200),window_size=15)**  
  Retourne un environnement gym  
> `df` : dataframe utilisé ultériement pour l'entrainement  
> `frame_bound` : partie du dataset utilisé pour l'entrainement  
> `window_size`: taille de l'observation par l'agent  


* **entrainement(env,algo=A2C,timesteps=50000)**  
  Retourne un modèle entrainé sur un environnement  
> `env` : environnement sur lequel l'entrainement va se réaliser  
> `algo` : algo de RL utilisé, (PPO, A2C ou DQN)  
> `timestep`: nombre d'étapes d'entrainement  

* **evaluation(df,model,frame_bound=(200,300),window_size=15)**  
  Permet d'évaluer le modèle sur un dataset  
> `df` : dataframe sur lequel l'evaluation va etre   
>  `model` : modèle qui va etre évalué  
> `frame_bound` : partie du dataset utilisé pour l'entrainement  
> `window_size`: taille de l'observation par l'agent  

* **graphique**
    Permet de tracer le graphique deschoix du modèle. 


# Reference 
[https://github.com/AminHP/gym-anytrading](https://github.com/AminHP/gym-anytrading)