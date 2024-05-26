# Module 4: agent.py
from itertools import count
import torch
import torch.optim as optim
import math
import random
import numpy as np
from connect_x import ConnectX
from replay_memory import ReplayMemory
from dqn_model import DQN
import torch.nn.functional as F
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
def random_agent(available_actions):
    return random.choice(available_actions)

class Agent:
    def __init__(self):
        self.env = ConnectX()
        self.memory = ReplayMemory()
        self.policy_net = DQN(self.env.board_width).to(device)
        self.target_net = DQN(self.env.board_width).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.steps_done = 0
        self.NUM_EPISODES = 20000
        self.BATCH_SIZE = 256
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 2000
        self.TARGET_UPDATE = 10

    def select_action(self, state, available_actions, training=True):
        state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
        epsilon = random.random()
        if training:
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1 * self.steps_done / self.EPS_DECAY)
        else:
            eps_threshold = 0
        
        if epsilon > eps_threshold:
            with torch.no_grad():
                r_actions = self.policy_net(state)[0, :]
                state_action_values = [r_actions[action] for action in available_actions]
                argmax_action = np.argmax(state_action_values)
                greedy_action = available_actions[argmax_action]
                return greedy_action
        else:
            return random.choice(available_actions)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        state_batch, action_batch, reward_batch, next_state_batch = zip(*[(np.expand_dims(m[0], axis=0), \
                                            [m[1]], m[2], np.expand_dims(m[3], axis=0)) for m in transitions])
        state_batch = np.array(state_batch, dtype=np.float32)
        state_batch = torch.tensor(state_batch, device=device)
        
        action_batch = torch.tensor(action_batch, dtype=torch.long, device=device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float, device=device)
        non_final_mask = torch.tensor(tuple(map(lambda s_: s_[0] is not None, next_state_batch)), device=device)
        non_final_next_state = torch.cat([torch.tensor(s_, dtype=torch.float, device=device).unsqueeze(0) for s_ in next_state_batch if s_[0] is not None])
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_state).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def train(self):
        for i in tqdm(range(self.NUM_EPISODES), desc='Training'): 
            self.env.reset()
            state_p1 = self.env.board_state.copy()

            for t in count():
                available_actions = self.env.get_available_actions()
                action_p1 = self.select_action(state_p1, available_actions)
                self.steps_done += 1
                state_p1_, reward_p1 = self.env.make_move(action_p1, 'p1')

                if self.env.isDone:
                    if reward_p1 == 1:
                        self.memory.dump([state_p1, action_p1, 1, None])
                    else:
                        self.memory.dump([state_p1, action_p1, 0.5, None])
                    break

                available_actions = self.env.get_available_actions()
                action_p2 = random_agent(available_actions)
                state_p2_, reward_p2 = self.env.make_move(action_p2, 'p2')

                if self.env.isDone:
                    if reward_p2 == 1:
                        self.memory.dump([state_p1, action_p1, -1, None])
                    else:
                        self.memory.dump([state_p1, action_p1, 0.5, None])
                    break

                self.memory.dump([state_p1, action_p1, -0.05, state_p2_])
                state_p1 = state_p2_

                self.optimize_model()

            if i % self.TARGET_UPDATE == self.TARGET_UPDATE - 1:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        print('Training complete')

    def demo(self):
        self.env.reset()
        self.env.render()

        while not self.env.isDone:
            state = self.env.board_state.copy()
            available_actions = self.env.get_available_actions()
            action = self.select_action(state, available_actions, training=False)
            state, reward = self.env.make_move(action, 'p1')
            self.env.render()

            if reward == 1:
                break

            available_actions = self.env.get_available_actions()
            action = random_agent(available_actions)
            state, reward = self.env.make_move(action, 'p2')
            self.env.render()

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
        
    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.policy_net.eval()