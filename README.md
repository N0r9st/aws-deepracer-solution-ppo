# AWS Deepracer Simulation round solution

Solution was developed in collaboration with @DAOvb

Solution is based on a simple PPO algorithm, achieves near-optimal score in AWS Deepracer environment.

To start training run:
```
docker-compose up -d
python run_ppo_train.py
```

To create a gif of agent's rollout run
```
python create_gif.py save/best_416_iconic-field-98_6env_SCLight_4st1sk.pickle 
```
![Example gif](gif.gif)