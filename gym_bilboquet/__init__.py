from gym.envs.registration import register

register(
    id='bilboquet-v0',
    entry_point='gym_bilboquet.envs:GameAI'
)