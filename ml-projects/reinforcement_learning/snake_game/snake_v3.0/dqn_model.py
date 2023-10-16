import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Input

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=12000)
        self.memory_len = 0
        self.min_replay_size = 500
        self.replay_frequency = 5
        self.batch_size = 200
        self.gamma = 0.9
        self.epsilon = 1
        self.epsilon_decay = 0.96
        self.epsilon_min = 0.01
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model_update_frequency = 50
        self.update_target_model()
    
    def build_model(self):
        model = Sequential([
            Input(shape=self.state_size),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ]) 
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.action_size)
        return np.argmax(self.model.predict(np.expand_dims(state, axis=0))[0])
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.memory_len += 1

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def replay(self):
        if self.memory_len < self.min_replay_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([experience[0] for experience in minibatch])
        qs_list = self.model.predict(states)

        next_states = np.array([experience[3] for experience in minibatch])
        next_qs_list = self.target_model.predict(next_states)

        X, y = [], []
        
        for index, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target += self.gamma * np.amax(next_qs_list[index])
            
            qs = qs_list[index]
            qs[action] = target

            X.append(state)
            y.append(qs)
        
        self.model.fit(np.array(X), np.array(y), batch_size=self.batch_size, verbose=0, epochs=1)

        self.decay_epsilon()