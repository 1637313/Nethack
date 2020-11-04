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
from torch.distributions import Categorical
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.model = PolicyValueNetwork(action_space).to(device)

        # for example, if your agent had a Pytorch model it must be load here
        self.model.load_state_dict(torch.load( 'models/actor_critic_nethack.pth', map_location=torch.device(device)))

    def act(self, observation):
        # Perform processing to observation

        # TODO: return selected action
        
        glyphs_matrix = observation['glyphs']
        agent_stat = observation['blstats']
        agent_row = observation['blstats'][1]
        agent_col = observation['blstats'][0]

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
            dist, value = self.model(state)
            action = dist.sample().item()
        return action_to_mean(action)

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

    save_dir = './animations_actor_critic/'
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


class PolicyValueNetwork(nn.Module):
    def __init__(self,action_space):
        super(PolicyValueNetwork, self).__init__()
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
        self.policy = nn.Linear(128,action_space)
        self.state_value = nn.Linear(128,1)

    def _select(self,embed,x):
        output = embed.weight.index_select(0,x.reshape(-1))
        return output.reshape(x.shape+(-1,))
    
    def forward(self, state):
        x = self._select(self.embed1,state[0])
        x = x.permute(0,3,1,2)
        x = self.glyph_model(x)
        x = torch.reshape(x,(x.size(0),-1))
        
        y = self._select(self.embed1,state[1])
        y = y.permute(0,3,1,2)
        y = self.around_agent_model(y)
        y = torch.reshape(y,(y.size(0),-1))
        
        z = self.agent_stats_mlp(state[2].float())
        z = torch.reshape(z,(z.size(0),-1))
        
        o = torch.cat((x, y, z), 1)
        o_t = self.mlp_o(o)
        policy_logits = self.policy(o_t)
        state_value = self.state_value(o_t)
        prob = F.softmax(policy_logits,dim=1)
        dist = Categorical(prob)
        return dist, state_value
    
def compute_returns(rewards, gamma):
    returns = []
    g = 0
    for reward in reversed(rewards):
        g = reward + gamma * g
        returns.insert(0, g)
    returns = np.array(returns)
    mu = returns.mean()
    std= returns.std()
    returns = (returns-mu)/std
    return returns.copy()



def generate_episode(policy_model,env):
    step =0
    done = False
    state_value = []
    rewards = []
    log_prob_actions = []
    state = env.reset()
    
    while (not done):        
        glyphs_matrix,around_agent,agent_stat = to_state(state)
        
        glyphs_matrix = torch.from_numpy(glyphs_matrix).type(torch.LongTensor).to(device).unsqueeze(0)
        agent_stat = torch.from_numpy(agent_stat).to(device).unsqueeze(0)
        around_agent = torch.from_numpy(around_agent).type(torch.LongTensor).to(device).unsqueeze(0)
        state = (glyphs_matrix,around_agent,agent_stat)
        
        dist,value = policy_model(state)
        action = dist.sample()
        
        log_p = dist.log_prob(action)
        state,reward,done,_ = env.step(action_to_mean(action))
        
        # clip rewards
        reward = np.tanh(reward/100)
        
        step +=1
        log_prob_actions.append(log_p)
        rewards.append(reward)
        state_value.append(value)
    return state_value.copy(),log_prob_actions.copy(),rewards.copy()

def reinforce_learned_baseline(env, policy_model, seed, learning_rate,
                               number_episodes,
                               gamma, verbose=False):
    # set random seeds (for reproducibility)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    env.seed(seed)
    random.seed(seed)
    
    optimizer = optim.RMSprop(policy_model.parameters(),lr=learning_rate,eps=0.000001)
    
    scores = []
    
    for episode in range(number_episodes):
        state_value,log_prob_actions,rewards = generate_episode(policy_model,env)
        scores.append(sum(rewards))

        returns = torch.from_numpy(compute_returns(rewards, gamma)).to(device)
        
        state_value = torch.cat(state_value)
        log_prob_actions = torch.cat(log_prob_actions)

        change = returns - state_value
        p_loss = -torch.sum(log_prob_actions*change.detach())
        v_loss = 0.5*torch.sum(change**2)
        loss = p_loss + v_loss

        optimizer.zero_grad()
        loss.backward()
        
        # gradient norm clipping of 40
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 40)
        optimizer.step()
    PATH = 'models/actor_critic_nethack.pth'
    torch.save(policy_model.state_dict(), PATH)   
    return policy_model,PATH, scores.copy()

def main():
    # Seed
    seeds = [1,2,3,4,5]

    # Initialise environment
    env = gym.make("NetHackScore-v0")
    
    # hyper-parameters
    gamma = 0.99
    learning_rate = 0.02
    # seed = 214
    seed = np.random.choice(seeds)
    number_episodes = 1250
    policy_model = PolicyValueNetwork(19)
    policy_model.to(device)

    net,path, scores = reinforce_learned_baseline(env, policy_model, seed, learning_rate,
                                             number_episodes,
                                             gamma, verbose=True)
    
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
    
    plt.plot(scores)
    plt.ylabel("score")
    plt.xlabel("episode")
    plt.savefig('img/ac_score_nethack.png')


if __name__ == '__main__':
    main()
