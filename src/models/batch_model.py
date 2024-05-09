import math, torch, torchvision
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn

CROP_DIMS = 60, 40, 95, 95
RESIZE = 120, 200

def phi(observation, device):
    x = observation.transpose([2, 0, 1])
    x = torch.tensor(x, dtype=torch.float32, device=device)
    # x = torchvision.transforms.functional.crop(x, *CROP_DIMS)
    # x = torchvision.transforms.Grayscale()(x)
    x = torchvision.transforms.Resize(RESIZE, antialias=True)(x)
    mean, std = x.mean([1,2]), x.std([1,2])
    x = torchvision.transforms.Normalize(mean=mean, std=std)(x)
    return x

def prompt_conv(x):
    print(x.shape)
    x = x[-1]
    if (len(x)) == 1 : return
    x = x.cpu().numpy()
    n = len(x)
    n = math.ceil(math.sqrt(n))
    f, axarr = plt.subplots(n,n, gridspec_kw={'wspace': 0, 'hspace': 0}, dpi=200)
    for i, img in enumerate(x):
        axarr[i // n, i % n].imshow(img, cmap='gray')
        
    for ax in axarr.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()

def batch_images(batch):
    grid = torchvision.utils.make_grid(batch)
    grid = grid.cpu().numpy()
    plt.imshow(grid.transpose([1,2,0]))
    plt.show()
    
def forwad_batch_images(batch):
    batch = batch.unsqueeze(0)
    batch = batch.transpose(0,1)
    grid = torchvision.utils.make_grid(batch)
    grid = grid.cpu().detach().numpy()
    plt.imshow(grid.transpose([1,2,0]))
    plt.show()

"""
camera.set_foreground_game()
observation = game_env.reset()
print(observation.shape)
x = phi(observation).cpu().numpy().transpose([1,2,0])
print(x.shape)
plt.imshow(x)
plt.show()
"""

class DQN(nn.Module):
    def __init__(self, action_space, lstm_n, lstm_layers):
        super(DQN, self).__init__()
        # 3x40x40

        self.conv_0 = nn.Conv2d(3, 16, (5, 5), stride=2) # 3x37x37
        self.relu_0 = nn.LeakyReLU()
        self.bn_0 = nn.BatchNorm2d(16)
        self.maxpool_0 = nn.MaxPool2d(2, stride=2)

        self.conv_1 = nn.Conv2d(16, 32, (5, 5), stride=2) # 4x6x6
        self.relu_1 = nn.LeakyReLU()
        self.bn_1 = nn.BatchNorm2d(32)
        self.maxpool_1 = nn.MaxPool2d(2, stride=2)

        self.conv_2 = nn.Conv2d(32, 32, (5, 5), stride=2) # 4x6x6
        self.relu_2 = nn.LeakyReLU()
        self.bn_2 = nn.BatchNorm2d(32)
        self.maxpool_2 = nn.MaxPool2d(2, stride=2)

        self.conv_3 = nn.Conv2d(32, 32, (5, 5), stride=2) # 4x6x6
        self.relu_3 = nn.LeakyReLU()
        self.bn_3 = nn.BatchNorm2d(32)
        self.maxpool_3 = nn.MaxPool2d(2, stride=2)

        self.flatten = nn.Flatten() # 4x6x6
        
        self.lstm = nn.LSTM(2112, lstm_n, lstm_layers, batch_first=True)
        self.relu_lstm = nn.LeakyReLU()
        
        self.v = nn.Linear(lstm_n, 1)
        self.a = nn.Linear(lstm_n, action_space)
        
    def forward(self, x, hn_cn):
        (hn, cn) = hn_cn
        #batch_size = x.shape[0]
        # x = x.reshape((-1, x.shape[-3], x.shape[-2], x.shape[-1]))
        x = self.conv_0(x)
        x = self.relu_0(x)
        x = self.bn_0(x)
        x = self.maxpool_0(x)
        
        x = self.conv_1(x)
        x = self.relu_2(x)
        x = self.bn_1(x)
        x = self.maxpool_1(x)

        #x = self.conv_2(x)
        #x = self.relu_1(x)
        #x = self.bn_2(x)
        #x = self.maxpool_2(x)

        #x = self.conv_3(x)
        #x = self.relu_3(x)
        #x = self.bn_3(x)
        #x = self.maxpool_3(x)

        # x = x.reshape((batch_size, -1, x.shape[-3], x.shape[-2], x.shape[-1]))
        x = torch.flatten(x, 1)
        #x = F.relu(self.fc1(x))
        x, (hn, cn) = self.lstm(x, (hn, cn))
        x = self.relu_lstm(x)
        # x = x[:,-1,:]

        a = self.a(x)
        v = self.v(x)
        q = v + a - a.mean()
        return q, (hn, cn)
    
    def forward_prompt(self, x, hn_cn):
        (hn, cn) = hn_cn
        #batch_size = x.shape[0]
        # x = x.reshape((-1, x.shape[-3], x.shape[-2], x.shape[-1]))
        x = self.conv_0(x)
        prompt_conv(x)
        x = self.bn_0(x)
        prompt_conv(x)
        #x = self.maxpool_0(x)
        
        x = self.conv_1(x)
        prompt_conv(x)
        x = self.bn_1(x)
        prompt_conv(x)
        #x = self.maxpool_1(x)

        x = self.conv_2(x)
        prompt_conv(x)
        x = self.bn_2(x)
        prompt_conv(x)
        #x = self.maxpool_2(x)

        x = self.conv_3(x)
        prompt_conv(x)
        x = self.bn_3(x)
        prompt_conv(x)
        #x = self.maxpool_3(x)

        # x = x.reshape((batch_size, -1, x.shape[-3], x.shape[-2], x.shape[-1]))
        x = torch.flatten(x, 1)
        print(x.shape)
        #x = F.relu(self.fc1(x))
        x, (hn, cn) = self.lstm(x, (hn, cn))
        print(x.shape)
        
        # x = x[:,-1,:]

        a = self.a(x)
        v = self.v(x)
        q = v + a - a.mean()
        input("continue...")
        return q, hn, cn
    
    
    def q_train(self, target_net, optimizer, loss_fn, sequence, gamma, lstm_n, lstm_layers, device):
        states, actions, rewards, next_states = *zip(*sequence), # let the ',' to not give syntax error
        sequence_length = len(states) # - 1

        states = torch.stack(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)

        non_final_states_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), device=device)
        non_final_next_states = torch.stack([s for s in next_states if s is not None])
        
        hn = torch.zeros(lstm_layers, lstm_n, dtype=torch.float32, device=device)
        cn = torch.zeros(lstm_layers, lstm_n, dtype=torch.float32, device=device)
        max_action_qvalues = torch.zeros(sequence_length, device=device)
        with torch.no_grad():
            y, (hn, cn) = target_net(non_final_next_states, (hn, cn))
            y_max = y.max(1)[0]
            max_action_qvalues[non_final_states_mask] = y_max
            
        # Set yj for terminal and non-terminal phij+1
        y = rewards + gamma * max_action_qvalues
        
        hn = torch.zeros(lstm_layers, lstm_n, dtype=torch.float32, device=device, requires_grad=True)
        cn = torch.zeros(lstm_layers, lstm_n, dtype=torch.float32, device=device, requires_grad=True)

        optimizer.zero_grad()
        qvalues, (hn, cn) = self(states, (hn, cn))

        qvalues = qvalues.gather(1, actions)
        loss = loss_fn(qvalues, y.unsqueeze(1))
        
        loss.backward()
        
        torch.nn.utils.clip_grad_value_(self.parameters(), 100)
        optimizer.step()
        
        return loss