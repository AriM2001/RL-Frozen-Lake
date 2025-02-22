"""
runner.py

This script runs the FrozenLake environment (using the gymnasium package) and interacts with the RLAgent
to train using Q-learning and SARSA algorithms. Training episodes run without rendering, and after training,
a demonstration episode is executed with human rendering.
"""

import gymnasium as gym
import time
from rl_agent import RLAgent

def run_episode(env, agent, method='q_learning', training=True, render=False):
    """
    Run a single episode in the environment with the given agent.
    
    Args:
        env: The FrozenLake environment.
        agent: Instance of RLAgent.
        method (str): 'q_learning' or 'sarsa'.
        training (bool): Whether to update the Q-table during the episode.
        render (bool): Whether to render the environment.
    
    Returns:
        total_reward (float): Cumulative reward obtained in the episode.
        steps (int): Number of steps taken.
    """
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    # For SARSA, pre-select an action.
    if method == 'sarsa':
        action = agent.choose_action(state, method)

    while not done:
        if method == 'q_learning':
            action = agent.choose_action(state, method)
        
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward

        if training:
            if method == 'q_learning':
                agent.learn_q_learning(state, action, reward, next_state, done or truncated)
            elif method == 'sarsa':
                next_action = agent.choose_action(next_state, method)
                agent.learn_sarsa(state, action, reward, next_state, next_action, done or truncated)
                action = next_action
        
        state = next_state
        steps += 1
        
        if render:
            env.render()
            # time.sleep(0.1)
        
        if done or truncated:
            break

    return total_reward, steps

def main():
    training_episodes = 10000
    
    # Create environment for training (no rendering)
    env = gym.make("FrozenLake-v1", is_slippery=True)
    state_space_size = env.observation_space.n
    action_space_size = env.action_space.n

    # Instantiate RL agents for Q-learning and SARSA.
    q_agent = RLAgent(state_space_size, action_space_size)
    sarsa_agent = RLAgent(state_space_size, action_space_size)

    print("Training with Q-learning:")
    for episode in range(training_episodes):
        total_reward, steps = run_episode(env, q_agent, method='q_learning', training=True, render=False)
        q_agent.update_exploration_rate()
        if (episode + 1) % 100 == 0:
            print(f"Q-learning Episode {episode + 1}: Reward: {total_reward}, Steps: {steps}")

    print("Training with SARSA:")
    for episode in range(training_episodes):
        total_reward, steps = run_episode(env, sarsa_agent, method='sarsa', training=True, render=False)
        sarsa_agent.update_exploration_rate()
        if (episode + 1) % 100 == 0:
            print(f"SARSA Episode {episode + 1}: Reward: {total_reward}, Steps: {steps}")
    
    env.close()

    # Demonstration episode with human rendering after training
    print("Demonstration episode with Q-learning agent :")
    q_agent.epsilon = 0.0
    demo_env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")
    run_episode(demo_env, q_agent, method='q_learning', training=False, render=True)
    demo_env.close()

    print("Demonstration episode with SARSA agent :")
    sarsa_agent.epsilon = 0.0
    demo_env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")
    run_episode(demo_env, sarsa_agent, method='sarsa', training=False, render=True)
    demo_env.close()

if __name__ == "__main__":
    main()