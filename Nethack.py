import os
import matplotlib as mp
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mp.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import gym
import nle
import random
from gym import spaces

from collections import deque
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""replay buffer"""
class ReplayBuffer:
    """
    Simple storage for transitions from an environment.
    """

    def __init__(self, size):
        """
        Initialise a buffer of a given size for storing transitions
        :param size: the maximum number of transitions that can be stored
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, matrix,around_agent,agent_stat, action, reward,matrix_p,around_agent_p,agent_stat_p, done):
        """
        Add a transition to the buffer. Old transitions will be overwritten if the buffer is full.
        :param state: the agent's initial state
        :param action: the action taken by the agent
        :param reward: the reward the agent received
        :param next_state: the subsequent state
        :param done: whether the episode terminated
        """
        data = (matrix,around_agent,agent_stat, action, reward, matrix_p,around_agent_p,agent_stat_p, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, indices):
        matrixs,around_agents,agent_stats, actions, rewards, matrix_ps,around_agent_ps,agent_stat_ps, dones = [],[],[], [], [], [], [],[],[]
        for i in indices:
            data = self._storage[i]
            matrix,around_agent,agent_stat, action, reward, matrix_p,around_agent_p,agent_stat_p, done = data
            matrixs.append(np.array(matrix, copy=False))
            around_agents.append(np.array(around_agent, copy=False))
            agent_stats.append(np.array(agent_stat, copy=False))
            actions.append(action)
            rewards.append(reward)
            matrix_ps.append(np.array(matrix_p, copy=False))
            around_agent_ps.append(np.array(around_agent_p, copy=False))
            agent_stat_ps.append(np.array(agent_stat_p, copy=False))
            dones.append(done)
        return (
            np.array(matrixs),
            np.array(around_agents),
            np.array(agent_stats),
            np.array(actions),
            np.array(rewards),
            np.array(matrix_ps),
            np.array(around_agent_ps),
            np.array(agent_stat_ps),
            np.array(dones),
        )

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions from the buffer.
        :param batch_size: the number of transitions to sample
        :return: a mini-batch of sampled transitions
        """
        indices = np.random.randint(0, len(self._storage) - 1, size=batch_size)
        return self._encode_sample(indices)


class AbstractAgent:
    """
    AbstractAgent

    """
    def __init__(self):
        raise NotImplementedError()

    def act(self, observation):
        raise NotImplementedError()


class MyAgent(AbstractAgent):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        # TODO Initialise your agent's models
        self.model = DQN(observation_space,action_space).to(device)

        # for example, if your agent had a Pytorch model it must be load here
        self.model.load_state_dict(torch.load( 'models/dqn_nethack.pth', map_location=torch.device(device)))

    def act(self, observation):
        # Perform processing to observation

        # TODO: return selected action
        
        glyphs_matrix = state['glyphs']
        agent_stat = state['blstats']
        agent_row = state['blstats'][1]
        agent_col = state['blstats'][0]

        around_agent = np.zeros([9,9])
        row = agent_row - 4
        col = agent_col -4
        for i in range(9):
            for j in range(9):
                if row>0 and row<glyphs_matrix.shape[0] and col>0 and col<glyphs_matrix.shape[1]:
                    around_agent[i][j] = glyphs_matrix[row][col]
                col+=1
            col = agent_col -4
            row+=1
            
        glyphs_matrix = torch.from_numpy(glyphs_matrix).type(torch.LongTensor).to(device).unsqueeze(0)
        agent_stat = torch.from_numpy(agent_stat).to(device).unsqueeze(0)
        around_agent = torch.from_numpy(around_agent).type(torch.LongTensor).to(device).unsqueeze(0)
        state = (glyphs_matrix,around_agent,agent_stat)

        with torch.no_grad():
            outputs = self.model(state)
            action = torch.argmax(outputs)
        return action_to_mean(action.item())

class RandomAgent(AbstractAgent):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()


def run_episode(env):
    # create instance of MyAgent
    # from MyAgent import MyAgent
    agent = MyAgent(env.observation_space, 19)

    done = False
    episode_return = 0.0
    state = env.reset()

    save_dir = './animations/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    try:
        env = gym.wrappers.Monitor(
        env, save_dir, video_callable=lambda episode_id: True)
    except gym.error.Error as e:
        print(e)

    while not done:
        # pass state to agent and let agent decide action
        action = agent.act(state)
        env.render()
        new_state, reward, done, _ = env.step(action)
        episode_return += reward
        state = new_state
    return episode_return


"""model"""

class DQN(nn.Module):
    def __init__(self, observation_space: spaces.Box, action_space):
        """
        Initialise the DQN
        :param observation_space: the state space of the environment
        :param action_space: the action space of the environment
        """
        super(DQN,self).__init__()
        self.embed1 = nn.Embedding(5991,32)
        
        self.glyph_model = nn.Sequential(
                  nn.Conv2d(32, 16, kernel_size=3, stride=1,padding = 1),
                  nn.ReLU(),
                  nn.Conv2d(16, 16, kernel_size=3, stride=1,padding = 1),
                  nn.ReLU(),
                  nn.Conv2d(16, 16, kernel_size=3, stride=1,padding = 1),
                  nn.ReLU(),
                  nn.Conv2d(16, 16, kernel_size=3, stride=1,padding = 1),
                  nn.ReLU(),
                  nn.Conv2d(16, 16, kernel_size=3, stride=1,padding = 1),
                  nn.ReLU(),
                )

        self.around_agent_model = nn.Sequential(
                  nn.Conv2d(32, 16, kernel_size=3, stride=1,padding = 1),
                  nn.ReLU(),
                  nn.Conv2d(16, 16, kernel_size=3, stride=1,padding = 1),
                  nn.ReLU(),
                  nn.Conv2d(16, 16, kernel_size=3, stride=1,padding = 1),
                  nn.ReLU(),
                  nn.Conv2d(16, 16, kernel_size=3, stride=1,padding = 1),
                  nn.ReLU(),
                  nn.Conv2d(16, 16, kernel_size=3, stride=1,padding = 1),
                  nn.ReLU(),
                )
        
        self.agent_stats_mlp = nn.Sequential(
                  nn.Linear(25,32),
                  nn.ReLU(),
                  nn.Linear(32,32),
                  nn.ReLU(),
                )
        self.mlp_o = nn.Sequential(
                  nn.Linear(27872,128),
                  nn.ReLU(),
                  nn.Linear(128,128),
                  nn.ReLU(),
                )
        self.outputLayer = nn.Linear(128,action_space)

    def _select(self,embed,x):
        output = embed.weight.index_select(0,x.reshape(-1))
        return output.reshape(x.shape+(-1,))
    
    def forward(self, state):
        x = self._select(self.embed1,state[0])
        x = x.permute(0,3,1,2).contiguous()
        x = self.glyph_model(x)
        x = torch.reshape(x,(x.size(0),-1))
        
        y = self._select(self.embed1,state[1])
        y = y.permute(0,3,1,2).contiguous()
        y = self.around_agent_model(y)
        y = torch.reshape(y,(y.size(0),-1))
        
        z = self.agent_stats_mlp(state[2].float())
        z = torch.reshape(z,(z.size(0),-1))
        
        o = torch.cat((x, y, z), 1)
        o_t = self.mlp_o(o)
        out = self.outputLayer(o_t)
        return out

"""agent"""

class DQNAgent:
    def __init__(self,observation_space: spaces.Box,action_space,replay_buffer,lr,batch_size,gamma):
        """
        Initialise the DQN algorithm using the Adam optimiser
        :param action_space: the action space of the environment
        :param observation_space: the state space of the environment
        :param replay_buffer: storage for experience replay
        :param lr: the learning rate for Adam
        :param batch_size: the batch size
        :param gamma: the discount factor
        """
        self.action_space = action_space
        self.observation_space = observation_space
        self.dqn = DQN(observation_space,action_space)
        self.target = DQN(observation_space,action_space)
        self.criterion = nn.L1Loss()
        self.optimizer = optim.RMSprop(self.dqn.parameters(), lr=lr,eps=0.000001)
        self.dqn.to(device)
        self.target.to(device)
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_target_network()

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        matrixs,around_agents,agent_stats, actions, rewards, matrix_ps,around_agent_ps,agent_stat_ps, dones = self.replay_buffer.sample(self.batch_size)

        actions = torch.from_numpy(np.array(actions)).to(device)
        rewards = torch.from_numpy(np.array(rewards)).to(device)
        dones = torch.from_numpy(np.array(dones)).to(device)
        
        matrixs = torch.from_numpy(matrixs).type(torch.LongTensor).to(device)
        agent_stats = torch.from_numpy(agent_stats).to(device)
        around_agents = torch.from_numpy(around_agents).type(torch.LongTensor).to(device)
        states = (matrixs,around_agents,agent_stats)
        
        matrix_ps = torch.from_numpy(matrix_ps).type(torch.LongTensor).to(device)
        agent_stat_ps = torch.from_numpy(agent_stat_ps).to(device)
        around_agent_ps = torch.from_numpy(around_agent_ps).type(torch.LongTensor).to(device)
        next_states = (matrix_ps,around_agent_ps,agent_stat_ps)

        prediction = self.dqn(states)
        current_q_value = prediction.gather(1,actions.unsqueeze(1))

        target_prediction = self.target(next_states).detach()
        max_q,_ = torch.max(target_prediction,1)

        target_q_value = rewards + (dones*self.gamma*max_q)
        target_q_value = target_q_value.unsqueeze(1)
        
        self.optimizer.zero_grad()

        loss = self.criterion(current_q_value.float(),target_q_value.float())
        loss.backward()
        
        # gradient norm clipping of 40
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), 40)
        self.optimizer.step()

        return loss.item()
        
    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        # TODO update target_network parameters with policy_network parameters
        self.target.load_state_dict(self.dqn.state_dict())
    
    def act(self, state: np.ndarray):
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        """ 
        glyphs_matrix = state['glyphs']
        agent_stat = state['blstats']
        agent_row = state['blstats'][1]
        agent_col = state['blstats'][0]

        around_agent = np.zeros([9,9])
        row = agent_row - 4
        col = agent_col -4
        for i in range(9):
            for j in range(9):
                if row>0 and row<glyphs_matrix.shape[0] and col>0 and col<glyphs_matrix.shape[1]:
                    around_agent[i][j] = glyphs_matrix[row][col]
                col+=1
            col = agent_col -4
            row+=1
            
        glyphs_matrix = torch.from_numpy(glyphs_matrix).type(torch.LongTensor).to(device).unsqueeze(0)
        agent_stat = torch.from_numpy(agent_stat).to(device).unsqueeze(0)
        around_agent = torch.from_numpy(around_agent).type(torch.LongTensor).to(device).unsqueeze(0)
        state = (glyphs_matrix,around_agent,agent_stat)

        with torch.no_grad():
            outputs = self.dqn(state)
            action = torch.argmax(outputs)
        return action.item()
    
def to_state(state):
    glyphs_matrix = state['glyphs']
    agent_stat = state['blstats']
    agent_row = state['blstats'][1]
    agent_col = state['blstats'][0]

    around_agent = np.zeros([9,9])
    row = agent_row - 4
    col = agent_col -4
    for i in range(9):
        for j in range(9):
            if row>0 and row<glyphs_matrix.shape[0] and col>0 and col<glyphs_matrix.shape[1]:
                around_agent[i][j] = glyphs_matrix[row][col]
            col+=1
        col = agent_col -4
        row+=1
        
    return glyphs_matrix.copy(),around_agent.copy(),agent_stat.copy()
def action_to_mean(action):
    action_meaning = np.arange(1,17)
    action_meaning = np.append(action_meaning,20)
    action_meaning = np.append(action_meaning,21)
    action_meaning = np.append(action_meaning,22)
    return action_meaning[action]
    
def train_dqn(env,hyper_params):
    replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])
    agent=DQNAgent(env.observation_space,19,replay_buffer,\
                   hyper_params["learning-rate"],hyper_params["batch-size"],hyper_params["discount-factor"])

    eps_timesteps = hyper_params["eps-fraction"] * float(hyper_params["num-steps"])
    episode_rewards = [0.0]
    train_loss = []
    state = env.reset()
    
    for t in range(hyper_params["num-steps"]):
        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold=hyper_params["eps-start"]+fraction*(hyper_params["eps-end"]-hyper_params["eps-start"])
        sample = random.random()
        if sample <= eps_threshold:
            action = np.random.choice(19)
        else:
            action = agent.act(state)

        #take step in env
        state_prime,reward,done,_ = env.step(action_to_mean(action))
        
        # clip rewards
        reward = np.tanh(reward/100)
        
        matrix,around_agent,agent_stat = to_state(state)
        matrix_p,around_agent_p,agent_stat_p = to_state(state_prime)
        agent.replay_buffer.add(matrix,around_agent,agent_stat,action,reward,matrix_p,around_agent_p,agent_stat_p,float(not done))
        state = state_prime.copy()
        episode_rewards[-1] += reward
        if done:
            state = env.reset()
            episode_rewards.append(0.0)
        if(t>hyper_params["learning-starts"] and t%hyper_params["learning-freq"]==0):
            train_loss.append(agent.optimise_td_loss())
        if (t>hyper_params["learning-starts"] and t%hyper_params["target-update-freq"]==0):
            agent.update_target_network()
        if (t>hyper_params["learning-starts"] and len(episode_rewards)%hyper_params["print-freq"]==0):
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
    PATH = 'models/dqn_nethack.pth'
    torch.save(agent.dqn.state_dict(), PATH)
    return episode_rewards.copy(),mean_100ep_reward.copy(),train_loss.copy(),PATH

if __name__ == '__main__':
    # Seed
    seeds = [1,2,3,4,5]

    # Initialise environment
    env = gym.make("NetHackScore-v0")
    hyper_params = {
        "replay-buffer-size": 10000,  # replay buffer size
        "learning-rate": 0.0002,  # learning rate for RMSprob
        "discount-factor": 0.99,  # discount factor
        "num-steps": int(1e9),  # total number of steps to run the environment for
        "batch-size": 256,  # number of transitions to optimize at the same time
        "learning-starts": 10000,  # number of steps before learning starts
        "learning-freq": 5,  # number of iterations between every optimization step
        "use-double-dqn": True,  # use double deep Q-learning
        "target-update-freq": 1000,  # number of iterations between every target network update
        "eps-start": 1.0,  # e-greedy start threshold
        "eps-end": 0.01,  # e-greedy end threshold
        "eps-fraction": 0.1,  # fraction of num-steps
        "print-freq": 10000,
        "print-freq":10,
    }
    rewards,mean_rewards,loss,path = train_dqn(env,hyper_params)

    plt.figure(figsize=(15,15))
    plt.plot(rewards)
    plt.ylabel("reward")
    plt.xlabel("episode")
    plt.savefig('img/nethack_reward.png')
    
    plt.figure(figsize=(15,15))
    plt.plot(mean_rewards)
    plt.ylabel("mean 100 episode reward")
    plt.xlabel("episode")
    plt.savefig('img/nethack_mean_reward.png')
    
    plt.figure(figsize=(15,15))
    plt.plot(loss)
    plt.ylabel("loss")
    plt.savefig('img/nethack_loss.png')
    
    #Number of times each seed will be run
    num_runs = 10

    #Run a few episodes on each seed
    rewards = []
    for seed in seeds:
        env.seed(seed)
        seed_rewards = []
        for i in range(num_runs):
            seed_rewards.append(run_episode(env))
        rewards.append(np.mean(seed_rewards))

    # Close environment and print average reward
    env.close()
    print("Average Reward: %f" %(np.mean(rewards)))