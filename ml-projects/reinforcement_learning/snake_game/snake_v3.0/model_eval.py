import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')


class ModelEval:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def train_model(self, num_episodes=2000):
        rewards_collected = []
        rewards_chunk_size = self.agent.target_model_update_frequency
        high_score = 15

        for episode in range(1, num_episodes+1):
            state = self.env.reset()
            episode_reward = 0

            while not self.env.done:
                action = self.agent.act(state)
                reward, done = self.env.step(action)
                next_state = self.env.get_state()

                self.agent.remember(state, action, reward, next_state, done)
                episode_reward += reward

                state = next_state
            
            rewards_collected.append(episode_reward)
            
            if episode % self.agent.replay_frequency == 0:
                self.agent.replay()

            if self.env.score > high_score:
                high_score = self.env.score
                self.agent.model.save(f"models/snake_model_{high_score}.h5")
                
            print(f"Episode: {episode} Reward: {episode_reward} Score: {self.env.score}")
            
            # Update target model after every (50) episodes and show the rewards collected
            if episode % self.agent.target_model_update_frequency == 0:
                # Show avg reward collected for the last 50 episodes
                print(f"Last {rewards_chunk_size} rewards avg: {sum(rewards_collected[-rewards_chunk_size:])/rewards_chunk_size}")
                print(f"Epsilon: {self.agent.epsilon}")
                self.agent.update_target_model()
            
            # After each episode, clear the session to release memory
            tf.keras.backend.clear_session()
        
        # Plot the rewards progress
        moving_avg = np.convolve(rewards_collected, np.ones(rewards_chunk_size)/rewards_chunk_size, mode='valid')
        plt.plot([i for i in range(num_episodes-rewards_chunk_size+1)], moving_avg)
        plt.ylabel(f"{rewards_chunk_size} Rewards MA")
        plt.xlabel(f"Episode")
        plt.show()

    def test_model(self, num_episodes=50):
        self.agent.epsilon = 0
        high_score = 0
        total_score = 0
        refresh_frequency = 50

        for episode in range(1, num_episodes + 1):
            state = self.env.reset()
            step_count = 0

            while not self.env.done:
                self.env.render()
                # Choose actions greedily (exploit) based on the learned Q-values
                action = self.agent.act(state)
                reward, done = self.env.step(action)
                next_state = self.env.get_state()

                state = next_state
                step_count += 1
                if step_count % refresh_frequency == 0:
                    tf.keras.backend.clear_session()
            print(f"Score: {self.env.score}")
            if self.env.score > high_score:
                high_score= self.env.score
            total_score += self.env.score
            print(self.env.terminations)

        print(f"Avg score: {total_score // num_episodes}")

        cv2.destroyAllWindows()
