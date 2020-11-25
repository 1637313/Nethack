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
        self.model = actor_critic_lstm(action_space).to(device)

        # for example, if your agent had a Pytorch model it must be load here
        PATH = cwd+'/models/actor_critic_lstm.pth'
        self.model.load_state_dict(torch.load(PATH, map_location=torch.device(device)))
        self.hidden_state,self.cell_state = model.init_states()

    def act(self, observation):
        # Perform processing to observation

        # TODO: return selected action
        
        glyphs_matrix,around_agent,agent_stat = to_state(observation)
        glyphs_matrix = torch.from_numpy(glyphs_matrix).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)/5991.0
        agent_stat = torch.from_numpy(agent_stat).type(torch.FloatTensor).to(device).unsqueeze(0)
        around_agent = torch.from_numpy(around_agent).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)/5991.0
        state = (glyphs_matrix,around_agent,agent_stat)

        with torch.no_grad():
            dist,value,self.hidden_state,self.cell_state = model(state,self.hidden_state,self.cell_state)
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


"""model"""

class actor_critic_lstm(nn.Module):
    def __init__(self,
                 action_space,
                 lstm_input_size = 52,
                 lstm_seq = 52,
                 lstm_hidden_size = 128,
                 lstm_num_layers = 1,
                ):
        super(actor_critic_lstm,self).__init__()
        self.action_space = action_space
        self.lstm_input_size = lstm_input_size
        self.lstm_seq = lstm_seq
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.glyph_model = nn.Sequential(
                  nn.Conv2d(1, 16, kernel_size=3, stride=1,padding = 1),
                  nn.ReLU(),
                  nn.Conv2d(16, 32, kernel_size=3, stride=1,padding = 1),
                  nn.ReLU(),
                  nn.Conv2d(32, 64, kernel_size=3, stride=1,padding = 1),
                  nn.ReLU(),
                  nn.Conv2d(64, 64, kernel_size=3, stride=1,padding = 1),
                  nn.ReLU(),
                  nn.Conv2d(64, 64, kernel_size=3, stride=1,padding = 1),
                  nn.ReLU(),
                )

        self.around_agent_model = nn.Sequential(
                  nn.Conv2d(1, 16, kernel_size=3, stride=1,padding = 1),
                  nn.ReLU(),
                  nn.Conv2d(16, 32, kernel_size=3, stride=1,padding = 1),
                  nn.ReLU(),
                  nn.Conv2d(32, 64, kernel_size=3, stride=1,padding = 1),
                  nn.ReLU(),
                  nn.Conv2d(64, 64, kernel_size=3, stride=1,padding = 1),
                  nn.ReLU(),
                  nn.Conv2d(64, 64, kernel_size=3, stride=1,padding = 1),
                  nn.ReLU(),
                )
        
        self.agent_stats_mlp = nn.Sequential(
                  nn.Linear(25,32),
                  nn.ReLU(),
                  nn.Linear(32,32),
                  nn.ReLU(),
                )
        self.lstm = nn.LSTM(self.lstm_input_size,
                            self.lstm_hidden_size,
                            self.lstm_num_layers,
                            batch_first=True) #(Input,Hidden,Num Layers)
        self.mlp_o = nn.Sequential(
                  nn.Linear(111392,2704),
                  nn.ReLU(),
                )
        self.policy = nn.Linear(self.lstm_hidden_size*self.lstm_seq,
                                     self.action_space)
        self.state_value = nn.Linear(self.lstm_hidden_size*self.lstm_seq,1)

    
    def forward(self, state,hidden_state,cell_state,dones = None):
        batch_size = state[0].shape[0]
        x = self.glyph_model(state[0])
        x = torch.reshape(x,(x.size(0),-1))
        
        y = self.around_agent_model(state[1])
        y = torch.reshape(y,(y.size(0),-1))
        
        z = self.agent_stats_mlp(state[2])
        z = torch.reshape(z,(z.size(0),-1))
        
        o = torch.cat((x, y, z), 1)

        o_t = self.mlp_o(o)
 
        #LSTM
        h = o_t.view(batch_size,self.lstm_seq,
                     self.lstm_input_size)
        if dones == None:
            h,(hidden_state,cell_state) = self.lstm(h,(hidden_state,cell_state))
        else:
            output_list = []
            for input_state,nd in zip(h.unbind(),dones.unbind()):
                # Reset core state to zero whenever an episode ends
                # Make done broadcastable with (num_layers,batch,hidden_size)
                nd = nd.view(1,-1,1)
                out,(hidden_state,cell_state) =\
                self.lstm(input_state.unsqueeze(0),(nd*hidden_state,nd*cell_state))
                # h (batch_size,seq_len,hidden_size)
                output_list.append(out)
            h = torch.cat(output_list)# -> (batch_size,seq_len,hidden_size)
               
        #(batch_size,sequence_length,hidden_size) -> (batch,sequence*hidden)
        h=h.view(h.shape[0],-1)

        policy_logits = self.policy(h)
        state_value = self.state_value(h)
        prob = F.softmax(policy_logits,dim=-1)
        dist = Categorical(prob)
        return dist, state_value,hidden_state,cell_state  

    def init_states(self):
        batch_size = 1
        hidden_state = torch.zeros(self.lstm_num_layers,
                                   batch_size,
                                   self.lstm_hidden_size).to(device)
        cell_state = torch.zeros(self.lstm_num_layers,
                                 batch_size,
                                 self.lstm_hidden_size).to(device)
        return hidden_state,cell_state
    
    def reset_states(self,hidden_state,cell_state):
        hidden_state[:,:,:] = 0
        cell_state[:,:,:] = 0
        return hidden_state.detach(),cell_state.detach()

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
                               max_steps = int(1e8),
                               verbose=False):
    # set random seeds (for reproducibility)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    env.seed(seed)
    random.seed(seed)
    
    optimizer = optim.RMSprop(policy_model.parameters(),lr=learning_rate,eps=0.000001)
    
    scores = [0.0]
    current_step_number = 0
    state = env.reset()
    hidden_state,cell_state = policy_model.init_states()
    
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
     
            dist,value,hidden_state,cell_state = policy_model(state_input,hidden_state,cell_state)
            
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
                state = env.reset()
                hidden_state,cell_state = policy_model.reset_states(hidden_state,cell_state)
                scores.append(0.0)
        
        glyphs_matrix,around_agent,agent_stat = to_state(state)
        glyphs_matrix = torch.from_numpy(glyphs_matrix).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)/5991.0
        agent_stat = torch.from_numpy(agent_stat).type(torch.FloatTensor).to(device).unsqueeze(0)
        around_agent = torch.from_numpy(around_agent).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)/5991.0
        state_input = (glyphs_matrix,around_agent,agent_stat)

#         _,next_value = policy_model(state_input)
        _,next_value,hidden_state,cell_state = policy_model(state_input,hidden_state,cell_state)
        
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
    PATH = cwd+'/models/actor_critic_lstm.pth'
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
    policy_model = actor_critic_lstm(env.action_space.n).to(device)
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
#     print("Average Reward: %f" %(np.mean(rewards)))
    
    
    plt.figure(figsize=(15,15))
    plt.plot(rewards)
    plt.ylabel("reward")
    plt.xlabel("episode")
    plt.title("episode rewards testing")
    plt.savefig(cwd+'/img/actor_critic_lstm_testing.png')
    
    
    plt.figure(figsize=(15,15))
    plt.plot(scores)
    plt.ylabel("score")
    plt.xlabel("episode")
    plt.savefig(cwd+'/img/actor_critic_lstm_training.png')


if __name__ == '__main__':
    main()