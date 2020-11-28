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

cwd = os.getcwd()

# convert the observation from env to the -> glyphs_matrix,around_agent,agent_stat
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
        PATH = cwd+'/models/Nethackac.pth'
        self.model.load_state_dict(torch.load(PATH, map_location=torch.device(device)))

    def act(self, observation):
        # Perform processing to observation

        # TODO: return selected action
        
        glyphs_matrix,around_agent,agent_stat = to_state(observation)
        glyphs_matrix = torch.from_numpy(glyphs_matrix).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)/5991.0
        agent_stat = torch.from_numpy(agent_stat).type(torch.FloatTensor).to(device).unsqueeze(0)
        around_agent = torch.from_numpy(around_agent).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)/5991.0
        state = (glyphs_matrix,around_agent,agent_stat)

        with torch.no_grad():
            dist, value = self.model(state)
            action = dist.sample().item()
        return action

class RandomAgent(AbstractAgent):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()


def run_episode(env):
    # create instance of MyAgent
    # from MyAgent import MyAgent
    agent = MyAgent(env.observation_space, env.action_space.n)

    done = False
    episode_return = 0.0
    state = env.reset()

    while not done:
        # pass state to agent and let agent decide action
        action = agent.act(state)
        new_state, reward, done, _ = env.step(action)
        episode_return += reward
        state = new_state
    return episode_return
 
class PolicyValueNetwork(nn.Module):
    def __init__(self,action_space):
        super(PolicyValueNetwork, self).__init__()
        self.glyph_model = nn.Sequential(
                  nn.Conv2d(1, 16, kernel_size=3, stride=1,padding = 1),
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
                  nn.Conv2d(1, 16, kernel_size=3, stride=1,padding = 1),
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

    
    def forward(self, state):
        x = self.glyph_model(state[0])
        x = torch.reshape(x,(x.size(0),-1))
        
        y = self.around_agent_model(state[1])
        y = torch.reshape(y,(y.size(0),-1))
        
        z = self.agent_stats_mlp(state[2])
        z = torch.reshape(z,(z.size(0),-1))
        
        o = torch.cat((x, y, z), 1)
        o_t = self.mlp_o(o)

        policy_logits = self.policy(o_t)
        state_value = self.state_value(o_t)
        prob = F.softmax(policy_logits,dim=-1)
        dist = Categorical(prob)
        return dist, state_value   

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def reinforce_learned_baseline(env, policy_model, seed, learning_rate,
                               number_episodes,
                               gamma,
                               num_step_td_update = 5,
                               max_steps = int(1e6),
                               verbose=False):
    # set random seeds (for reproducibility)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    env.seed(seed)
    random.seed(seed)
    
    optimizer = optim.RMSprop(policy_model.parameters(),lr=learning_rate,eps=0.000001)
    
    scores = [0.0]
    scores_mean = []
    current_step_number = 0
    state = env.reset()
    
    while current_step_number < max_steps:
        state_value = []
        rewards = []
        log_prob_actions = []
        masks = []

        for _ in range(num_step_td_update):
            
            
            glyphs_matrix,around_agent,agent_stat = to_state(state)
            glyphs_matrix = torch.from_numpy(glyphs_matrix).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)/5991.0
            agent_stat = torch.from_numpy(agent_stat).type(torch.FloatTensor).to(device).unsqueeze(0)
            around_agent = torch.from_numpy(around_agent).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)/5991.0
            state_input = (glyphs_matrix,around_agent,agent_stat)
     
            dist,value = policy_model(state_input)
            action = dist.sample()

            log_p = dist.log_prob(action)
            state_prime,reward,done,_ = env.step(action)
            
            scores[-1]+=reward

            # clip rewards
            reward = np.tanh(reward/100)

            log_prob_actions.append(log_p)
            state_value.append(value)
            masks.append(1 - done)
            rewards.append(reward)
            
            current_step_number += 1
            state = state_prime
            if done:
                if len(scores)%100==0:
                    scores_mean.append(round(np.mean(scores[-100:-1]), 1))
                state = env.reset()
                scores.append(0.0)
        
        glyphs_matrix,around_agent,agent_stat = to_state(state)
        glyphs_matrix = torch.from_numpy(glyphs_matrix).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)/5991.0
        agent_stat = torch.from_numpy(agent_stat).type(torch.FloatTensor).to(device).unsqueeze(0)
        around_agent = torch.from_numpy(around_agent).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)/5991.0
        state_input = (glyphs_matrix,around_agent,agent_stat)

        _,next_value = policy_model(state_input)
        
        returns = compute_returns(next_value, rewards, masks,gamma)
        
        state_value = torch.cat(state_value)
        log_prob_actions = torch.cat(log_prob_actions)
        returns   = torch.cat(returns).detach()

        change = returns - state_value
        p_loss = -torch.mean(log_prob_actions*change.detach())
        v_loss = 0.5*torch.mean(torch.pow(change,2))
        loss = p_loss + v_loss

        optimizer.zero_grad()
        loss.backward()
        
        # gradient norm clipping of 40
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 40)
        optimizer.step()
    PATH = cwd+'/models/Nethackac.pth'
    torch.save(policy_model.state_dict(), PATH) 
    
    return policy_model,PATH, scores.copy(),scores_mean.copy()

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
    policy_model = PolicyValueNetwork(env.action_space.n).to(device)
    net,path, scores,scores_mean = reinforce_learned_baseline(env, policy_model, seed, learning_rate,
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
#     print("Average Reward: %f" %(np.mean(rewards)))
    
    
    plt.figure(figsize=(15,15))
    plt.plot(rewards)
    plt.ylabel("reward")
    plt.xlabel("episode")
    plt.title("episode rewards testing")
    plt.savefig(cwd+'/img/actor_critic_testing_reward_nethackac.png')
    
    
    plt.figure(figsize=(15,15))
    plt.plot(scores)
    plt.ylabel("score")
    plt.xlabel("episode")
    plt.savefig(cwd+'/img/actor_critic_traing_reward_nethackac.png')
    
    plt.figure(figsize=(15,15))
    plt.plot(scores_mean)
    plt.ylabel("score(average over 100 eps)")
    plt.xlabel("episode")
    plt.savefig(cwd+'/img/actor_critic_traing_reward_nethackac.png')


if __name__ == '__main__':
    main()
