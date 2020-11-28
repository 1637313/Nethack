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
from torch.distributions import Categorical, Bernoulli
import random
from copy import deepcopy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cwd = os.getcwd()


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
        
        self.model = OptionCritic(action_space,8).to(device)

        # for example, if your agent had a Pytorch model it must be load here
        self.model.load_state_dict(torch.load( cwd+'/models/option_critic.pth', map_location=torch.device(device)))

    def act(self, observation):
        # Perform processing to observation

        # TODO: return selected action

        obs_tuple = tupleTensor(observation)

        with torch.no_grad():
            state = self.model.get_state(obs_tuple)
            greedy_option = self.model.greedy_option(state)
            dist, value = self.model(state)
            action ,_,_ = model.get_action(state,greedy_option)
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
    
class OptionCritic(nn.Module):
    def __init__(self,num_actions,num_options,temperature=1.0):
        super(OptionCritic,self).__init__()
        self.num_actions = num_actions
        self.num_options = num_options
        self.temperature = temperature
        
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
                  nn.modules.Flatten(),
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
                  nn.modules.Flatten(),
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
        
        self.state_option_value = nn.Linear(128,num_options)#policy_over_options
        self.terminations = nn.Linear(128,num_options)#option-termination
        self.options_w = nn.Parameter(torch.zeros(num_options,128,num_actions))
        self.options_b = nn.Parameter(torch.zeros(num_options,num_actions))
        
    def get_state(self,obs):
        x = self.glyph_model(obs[0])  
        y = self.around_agent_model(obs[1])
        z = self.agent_stats_mlp(obs[2])
        o = torch.cat((x, y, z), 1)
        state = self.mlp_o(o)
        return state
    
    def get_state_option_value(self,state):
        return self.state_option_value(state)
    
    def predict_option_termination(self,state,current_option):
        termination = self.terminations(state)[:,current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()
        
        state_option_value = self.get_state_option_value(state)
        next_option = state_option_value.argmax(dim=-1)
        return bool(option_termination.item()),next_option.item()
    
    def get_terminations(self,state):
        return self.terminations(state).sigmoid()
    
    def get_action(self,state,option):
        logits = state@self.options_w[option]+self.options_b[option]
        action_dist = (logits/self.temperature).softmax(dim=-1)
        action_dist = Categorical(action_dist)
        
        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        
        return action.item(),logp,entropy
    
    def greedy_option(self,state):
        state_option_value = self.get_state_option_value(state)
        return state_option_value.argmax(dim=-1).item()

def critic_loss(model,model_prime,data,gamma=0.99):
        matrixs,around_agents,agent_stats, options, rewards, matrix_ps,around_agent_ps,agent_stat_ps, dones = data
        
        batch_idx = torch.arange(len(options)).long()
        options = torch.from_numpy(np.array(options)).type(torch.LongTensor).to(device)
        rewards = torch.from_numpy(np.array(rewards)).type(torch.FloatTensor).to(device)
        masks = torch.from_numpy(np.array(dones)).type(torch.FloatTensor).to(device)
        
        matrixs = torch.from_numpy(matrixs).type(torch.FloatTensor).to(device).unsqueeze(1)/5991.0
        agent_stats = torch.from_numpy(agent_stats).type(torch.FloatTensor).to(device)
        around_agents = torch.from_numpy(around_agents).type(torch.FloatTensor).to(device).unsqueeze(1)/5991.0
        obs = (matrixs,around_agents,agent_stats)
        
        matrix_ps = torch.from_numpy(matrix_ps).type(torch.FloatTensor).to(device).unsqueeze(1)/5991.0
        agent_stat_ps = torch.from_numpy(agent_stat_ps).type(torch.FloatTensor).to(device)
        around_agent_ps = torch.from_numpy(around_agent_ps).type(torch.FloatTensor).to(device).unsqueeze(1)/5991.0
        next_obs = (matrix_ps,around_agent_ps,agent_stat_ps)
        
        #The loss is the TD loss of Q and the update target, so we need to calculate Q
        states = model.get_state(obs).squeeze(0)
        state_option_value = model.get_state_option_value(states)
        
        #the update target contains Q-next, but for stable learning we use prime network for this
        next_state_prime = model_prime.get_state(next_obs).squeeze(0)
        next_Q_prime = model_prime.get_state_option_value(next_state_prime).detach()
        
        # we need beta probabilities of the next state
        next_states = model.get_state(next_obs).squeeze(0)
        next_termination_probs = model.get_terminations(next_states).detach()
        next_options_term_prob = next_termination_probs[batch_idx,options]
        
        #calculate the update target gt
        gt = rewards+masks*gamma*((1-next_options_term_prob)*\
                                  next_Q_prime[batch_idx,options]+\
                                  next_options_term_prob*next_Q_prime.max(dim=-1)[0])
        
        td_err = (state_option_value[batch_idx,options] -\
                  gt.detach()).pow(2).mul(0.5).mean()
        return td_err
    
def actor_loss(obs,option,logp,entropy,reward,done,next_obs,model,model_prime,termination_reg=0.01,entropy_reg=0.01,gamma=0.99):
    state = model.get_state(obs)
    next_state = model.get_state(next_obs)
    next_state_prime = model_prime.get_state(next_obs)
    
    option_term_prob = model.get_terminations(state)[:,option]
    next_option_term_prob = model.get_terminations(next_state)[:,option].detach()
    
#     with torch.autograd.set_detect_anomaly(True):
    state_option_value = model.get_state_option_value(state).detach().squeeze()
    next_Q_prime = model_prime.get_state_option_value(next_state_prime).detach().squeeze()
    
    #Target update gt
    gt = reward+(1-done)*gamma*((1-next_option_term_prob)*next_Q_prime[option]+\
                                next_option_term_prob*next_Q_prime.max(dim=-1)[0])
    
    #The termination loss
    termination_loss = option_term_prob*(state_option_value[option].detach() - \
                                         state_option_value.max(dim=-1)[0].detach()+\
                                         termination_reg)*(1-done)
    
    # actor-critic policy gradient with entropy regularization
    policy_loss = -logp*(gt.detach()-state_option_value[option])-entropy_reg*entropy
    actor_loss = termination_loss+policy_loss
    return actor_loss

def tupleTensor(state):
    glyphs_matrix,around_agent,agent_stat = to_state(state)
    glyphs_matrix = torch.from_numpy(glyphs_matrix).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)/5991.0
    agent_stat = torch.from_numpy(agent_stat).type(torch.FloatTensor).to(device).unsqueeze(0)
    around_agent = torch.from_numpy(around_agent).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)/5991.0
    state_tuple = (glyphs_matrix,around_agent,agent_stat)
    return state_tuple


def train_optionCritic(env,hyper_params,num_options=8):
    
    replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])
    
    option_critic = OptionCritic(env.action_space.n,num_options).to(device)
    
    #create prime network for more stable Q values
    option_critic_prime = deepcopy(option_critic)
    
    optimizer = optim.RMSprop(option_critic.parameters(),lr=hyper_params["learning-rate"],\
                              eps=0.000001)
    
    eps_timesteps = hyper_params["eps-fraction"] * float(hyper_params["num-steps"])
    
    seed = 1
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)

    steps =0
    t=0
    episode_rewards = []
    while steps<hyper_params["num-steps"]:
        rewards = 0
        option_lengths = {option:[] for option in range(num_options)}
        
        obs = env.reset()
        obs_tuple = tupleTensor(obs)
        state = option_critic.get_state(obs_tuple)
        greedy_option = option_critic.greedy_option(state)
        current_option = 0
        
        done = False
        option_termination = True
        curr_op_len = 0
        ep_steps = 0
        while not done and ep_steps<2000:
            
            fraction = min(1.0, float(t) / eps_timesteps)
            eps_threshold=hyper_params["eps-start"]+fraction*(hyper_params["eps-end"]-hyper_params["eps-start"])
            t +=1
            
            
            if option_termination:
                option_lengths[current_option].append(curr_op_len)
                
                if random.random()<eps_threshold:
                    current_option = np.random.choice(num_options)
                else:
                    current_option = greedy_option
                curr_op_len =0
                
            action , logp, entropy = option_critic.get_action(state,current_option)
            
            next_obs, reward, done, _ = env.step(action)
            
            rewards+= reward
            
            next_obs_tuple = tupleTensor(next_obs)
            
            
            # clip rewards
            reward = np.tanh(reward/100)

            matrix,around_agent,agent_stat = to_state(obs)
            matrix_p,around_agent_p,agent_stat_p = to_state(next_obs)
            replay_buffer.add(matrix,around_agent,agent_stat,current_option,reward,matrix_p,around_agent_p,agent_stat_p,float(not done))
            
            option_termination,greedy_option = option_critic.predict_option_termination(\
                                                                                        option_critic.get_state(next_obs_tuple),current_option)
            
        
            if replay_buffer.__len__()>hyper_params["batch-size"]:
                loss = actor_loss(obs_tuple,current_option,logp,entropy,\
                                        reward,done,next_obs_tuple,option_critic,\
                                        option_critic_prime,gamma=hyper_params["discount-factor"])
               
                if steps%hyper_params["learning-freq"]==0:
                    data = replay_buffer.sample(hyper_params["batch-size"])
                    criticLoss = critic_loss(option_critic,option_critic_prime,data,gamma=hyper_params["discount-factor"])
                    loss += criticLoss
                    
                optimizer.zero_grad()
                
                loss.backward()
                optimizer.step()
                
                if steps%hyper_params["target-update-freq"]==0:
                    option_critic_prime.load_state_dict(option_critic.state_dict())
                    
            steps+=1
            ep_steps+=1
            curr_op_len +=1
            state = option_critic.get_state(next_obs_tuple)
            obs = next_obs
            obs_tuple = tupleTensor(obs)
            
        episode_rewards.append(rewards)
                                                                                                       
    PATH = cwd+'/models/option_critic.pth'
    torch.save(option_critic.state_dict(), PATH)
    return episode_rewards.copy(),PATH

if __name__ == '__main__':
    # Seed
    seeds = [1,2,3,4,5]

    # Initialise environment
    env = gym.make("NetHackScore-v0")
    hyper_params = {
        "replay-buffer-size": 10000,  # replay buffer size
        "learning-rate": 0.0002,  # learning rate for RMSprob
        "discount-factor": 0.99,  # discount factor
 #         "num-steps": int(1e7),
#         "num-steps": int(1e8),
#         "num-steps": int(1e9),
        "num-steps": int(1e6),  # total number of steps to run the environment for
        "batch-size": 32,  # number of transitions to optimize at the same time
        "learning-starts": 10000,  # number of steps before learning starts
        "learning-freq": 5,  # number of iterations between every optimization step
        "use-double-dqn": True,  # use double deep Q-learning
        "target-update-freq": 1000,  # number of iterations between every target network update
        "eps-start": 1.0,  # e-greedy start threshold
        "eps-end": 0.01,  # e-greedy end threshold
        "eps-fraction": 0.1,  # fraction of num-steps
        "print-freq":10,
    }
    train_rewards,path = train_optionCritic(env,hyper_params)
    
    plt.figure(figsize=(15,15))
    plt.plot(train_rewards)
    plt.ylabel("reward")
    plt.xlabel("episode")
    plt.title("episode rewards training")
    plt.savefig(cwd+'/img/option_critic_training.png')
    
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
    
    plt.figure(figsize=(15,15))
    plt.plot(rewards)
    plt.ylabel("reward")
    plt.xlabel("episode")
    plt.title("episode rewards testing")
    plt.savefig(cwd+'/img/option_critic_testing.png')
