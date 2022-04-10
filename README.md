Ce projet a été réalisé pendant mon cours de MTI880 pendant ma maitrise à l'ETS en Technologie de l'Information. 

Repris code de [Gym AnyTrading de AminHP](https://github.com/AminHP/gym-anytrading) en l'adaptant à notre cas, prédiction de stock. 


# Commande 
* **recuperation_donnee**  
  La fonction retourne un dataframe pandas avec comme index la date et comme colomne : "Open","High","Low","Close","Volume". 
> `name`: nom de l'action (nom utilisé dans le projet : *amzn, aapl, msft, goog, tsla, fb*)
> `start`: date début au format JJ/MM/YYYY
> `end`: date fin au format JJ/MM/YYYY
> `interval`: interval entre chaque mesure, 1d", "1wk", "1mo", or "1m" for daily, weekly, monthly, or minute data

* **creation_env**  
  Retourne un environnement gym
> `df` : dataframe utilisé ultériement pour l'entrainement
> `frame_bound` : partie du dataset utilisé pour l'entrainement
> `window_size`: taille de l'observation par l'agent
> 

* **entrainement**  
  Retourne un modèle entrainé sur un environnement
> `env` : environnement sur lequel l'entrainement va se réaliser
> `algo` : algo de RL utilisé, (PPO, A2C ou DQN)
> `timestep`: nombre d'étapes d'entrainement

* **evaluation**  
  Permet d'évaluer le modèle sur un dataset
> `df` : dataframe sur lequel l'evaluation va etre 
>  `model` : modèle qui va etre évalué
> `frame_bound` : partie du dataset utilisé pour l'entrainement
> `window_size`: taille de l'observation par l'agent

* **graphique**
    Permet de tracer le graphique deschoix du modèle. 


# Reference 
[https://github.com/AminHP/gym-anytrading](https://github.com/AminHP/gym-anytrading)