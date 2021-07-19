from gym.envs.registration import register

register(
    id='image_embedding-v0',
    entry_point='gym_image_embedding.envs:ImageEmbeddingEnv',
)
