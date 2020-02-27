import numpy as np
import torch
import torch.nn.functional as F
import isometry_init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def logits_info(obj, label, model):
    correct = 0
    logits, _ = model(obj)
    prob = F.softmax(logits, dim=1)

    rates, indices = prob.sort(1, descending=True) 
    rates, indices = rates.squeeze(0), indices.squeeze(0)  

    correct += indices[0].eq(label.data).cpu().sum()
    
    return logits, correct.item(), rates, indices



class environment():
    def __init__(self, d = 8, a0 = 0, b0 = 2*np.pi, train = False):
    
        self.d = d 
        self.a0 = a0
        self.b0 = b0
        self.generate_thetas()
        self.timestep = 0
        self.rewards = np.zeros(pow(d,3)).reshape(d,d,d) # n*n*n as n is the divisions of 2pi
        self.train = train
    
    def generate_thetas(self):
        self.thetas = np.random.uniform(0,1,pow(self.d,3)).reshape(self.d, self.d, self.d)
  
    def arm_to_interval(self, arm):
        a , b = np.zeros(3), np.zeros(3)
        for i in range(3):
            a[i] = self.a0 + (self.b0 - self.a0)*arm[i] / self.d
            b[i] = self.a0 + (self.b0 - self.a0)*(arm[i]+1) / self.d
        return a, b
  

    def get_reward_matrix(self, arm, obj, label, model):
        a, b = self.arm_to_interval(arm)
        
        matrix = isometry_init.reflection(a, b)
        model.iso.weight.data = torch.Tensor(matrix).to(device)
        _, correct, _, _ = logits_info(obj, label, model)
        if self.train:
            reward = correct # when training rewards from correct prediction
        else:
            reward = 1 - correct # when attack rewards from wrong predictionreward = 1 - correct # correct = 0 means prediction wrong, i.e. this interval is good for attack
        return reward, matrix


class BetaAlgo():

    def __init__(self, environment):
        self.environment = environment
        self.d = environment.d
        self.alpha = np.ones(pow(self.d,3)).reshape(self.d, self.d, self.d)
        self.beta = np.ones(pow(self.d,3)).reshape(self.d, self.d, self.d)
  
    def get_reward_matrix(self, arm, obj, label, model):
        reward, matrix = self.environment.get_reward_matrix(arm, obj, label, model)
        self._update_params(arm, reward)
        return reward, matrix


    def _update_params(self, arm, reward):
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward


class BernThompson(BetaAlgo):
    def __init__(self, environment):
        super().__init__(environment)
  
    def get_action(self):
        theta = np.random.beta(self.alpha, self.beta)
        indx = np.unravel_index(np.argmax(theta, axis=None), theta.shape)
        return indx