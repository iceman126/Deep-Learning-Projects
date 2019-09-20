# Deep Q-Network
[[images/Cartpole.gif | width=100px | alt=CartPole Demo Video]]
![CartPole Demo Video](images/Cartpole.gif)
![CartPole Evaluation Scores](images/Cartpole_eval.png)

## Usage
Training:
```
python dqn.py -env (env_name) -lr (learning_rate)
```

Evaluation:
```
python dqn.py -eval -cp (path/to/checkpoint)
```

## Future Work
- Customizable network structure
- More hyper-parameters can be specified through command line argument