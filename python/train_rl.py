import os
import sys
import numpy as np
import logging
import gym
from gym import envs
from torch.utils.tensorboard import SummaryWriter
import cv2 as cv

from agent.ppo import Agent


def train(env_name, train_visualize_fps):
    try:
        env = gym.make(env_name)
    except:
        logging.warning("please try run 'pip install -e gym_image_embedding' to register the environment")
        logging.warning("exception:", sys.exc_info())
        return

    n_steps_interval_learn = 20
    n_episodes = 10
    visualize = True
    model_path = "checkpoints"
    actor_model_name = "model_actor"
    critic_model_name = "model_critic"
    writer = SummaryWriter()

    batch_size = 5
    n_epochs = 4
    alpha = 0.0003

    checkpoint_path = model_path
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=env.observation_space.shape, chkpt_dir=checkpoint_path,
                  actor_chkpt_name=actor_model_name, critic_chkpt_name=critic_model_name, tensorboard_writer=writer)

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_episodes):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)

            if visualize:
                env.render()

                if train_visualize_fps > 0:
                    sleep_time = int(1. / train_visualize_fps * 1000.)  # in units of ms
                else:
                    sleep_time = 0

                key = cv.waitKey(sleep_time)

            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % n_steps_interval_learn == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_

        score_history.append(score)
        writer.add_scalar('Variable/score', score, i)

        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        logging.info('episode #%s has score %.3f with avg. score %.3f, steps %s, learning steps %s.' %
                     (i, score, avg_score, n_steps, learn_iters))

    # Note: this line below is necessary due to windows 10 error "KeyError weakref for Win32Window"
    env.unwrapped.viewer.close()
