from gym.envs.registration import register
register(entry_point='gym_example.envs:ExampleEnv',
         id='example-v0',
)