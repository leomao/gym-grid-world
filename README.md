Grid World Environments
===

Use [env_viewer][env_viewer] to view the environment.

## How to Use

```
import gym
import gym_grid_world

env = gym.make('pickput2d-v0')

# configure the view_name
env.configure(view_name='pickput')

# start to send observation to env_viewer
env.render()

# gym compatible
# obs = env.reset()
# obs, rew, done, info = env.step(action)

env.render(close=True)
```

## Environments

- Eat Bullet `eatbullet2d-v0`
- Pick Put `pickput2d-v0`

[env_viewer]: https://github.com/leomao/env_viewer
