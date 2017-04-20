Grid World Environments
===

## How to Use

```
import gym
import gym_grid_world

env = gym.make('pickput2d-v0')

# configure the enviroments
env.configure(...)

# start to send observation to env_viewer
env.render()

# gym compatible
# obs = env.reset()
# obs, rew, done, info = env.step(action)
```

## Environments

- Eat Bullet `eatbullet2d-v0`
- Pick Put `pickput2d-v0`
- Push Block `pushblock2d-v0`
