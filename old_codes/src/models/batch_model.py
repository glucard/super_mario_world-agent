import math, torch, torchvision, time, threading
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from IPython.display import clear_output

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

class PromptQValues:
    def __init__(self):
        self.q_values = []
        self.thread = threading.Thread(target=self.prompt_qvalues, daemon=True)
        self.thread.start()

    def set_qvalues(self, q_values):
        self.q_values = q_values

    def prompt_qvalues(self):
        while True:
            names = ["right", "left", "jump", "run"]
            # Figure Size
            # fig = plt.figure(figsize =(10, 7))
            # Horizontal Bar Plot
            if len(self.q_values) > 0:
                clear_output(wait=True)
                plt.bar(names[0:2], self.q_values)
                plt.ylim(-1, 1)
                self.q_values = []
                # Show Plot
                plt.show()
            time.sleep(0.3)

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

        self.conv_0 = nn.Conv2d(3, 16, (3, 3), padding=2, stride=1) # 3x37x37
        self.relu_0 = nn.LeakyReLU()
        self.bn_0 = nn.BatchNorm2d(16)
        self.maxpool_0 = nn.MaxPool2d(2, stride=2)

        self.conv_1 = nn.Conv2d(16, 32, (5, 5), padding=3, stride=1) # 4x6x6
        self.relu_1 = nn.LeakyReLU()
        self.bn_1 = nn.BatchNorm2d(32)
        self.maxpool_1 = nn.MaxPool2d(4, stride=4)

        self.conv_2 = nn.Conv2d(32, 32, (8, 8), padding=4, stride=4) # 4x6x6
        self.relu_2 = nn.LeakyReLU()
        self.bn_2 = nn.BatchNorm2d(32)
        self.maxpool_2 = nn.MaxPool2d(4, stride=4)

        self.conv_3 = nn.Conv2d(32, 32, (5, 5), stride=2) # 4x6x6
        self.relu_3 = nn.LeakyReLU()
        self.bn_3 = nn.BatchNorm2d(32)
        self.maxpool_3 = nn.MaxPool2d(2, stride=2)

        self.flatten = nn.Flatten() # 4x6x6
        
        self.lstm = nn.LSTM(3328, lstm_n, lstm_layers, batch_first=True)
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
        # x = self.maxpool_0(x)
        
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.bn_1(x)
        #x = self.maxpool_1(x)

        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.bn_2(x)
        x = self.maxpool_2(x)

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
        prompt_conv(x)
        (hn, cn) = hn_cn
        #batch_size = x.shape[0]
        # x = x.reshape((-1, x.shape[-3], x.shape[-2], x.shape[-1]))
        x = self.conv_0(x)
        x = self.relu_0(x)
        x = self.bn_0(x)
        # x = self.maxpool_0(x)
        prompt_conv(x)
        
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.bn_1(x)
        #x = self.maxpool_1(x)
        prompt_conv(x)

        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.bn_2(x)
        x = self.maxpool_2(x)
        prompt_conv(x)
        
        x = torch.flatten(x, 1)
        print(x.shape)
        x, (hn, cn) = self.lstm(x, (hn, cn))
        print(x.shape)
        x = self.relu_lstm(x)
        print(x.shape)

        a = self.a(x)
        v = self.v(x)
        q = v + a - a.mean()
        input("continue...")
        return q, hn, cn
    
    
    def q_train(self, target_net, optimizer, loss_fn, sample, gamma, lstm_n, lstm_layers, replay_memory, device):
        # states, actions, rewards, next_states, priorities, indices, weights = *zip(*sample), # let the ',' to not give syntax error
        states, actions, rewards, next_states, priorities, indices, weights = sample # let the ',' to not give syntax error
        sample_length = len(states) # - 1

        states = torch.stack(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        weights = torch.stack(weights)

        non_final_states_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), device=device)
        non_final_next_states = torch.stack([s for s in next_states if s is not None])
        
        hn = torch.zeros(lstm_layers, lstm_n, dtype=torch.float32, device=device)
        cn = torch.zeros(lstm_layers, lstm_n, dtype=torch.float32, device=device)
        max_action_qvalues = torch.zeros(sample_length, device=device)
        with torch.no_grad():
            y, (hn, cn) = target_net(non_final_next_states, (hn, cn))
            y_max = y.max(1)[0]
            max_action_qvalues[non_final_states_mask] = y_max
            
        # Set yj for terminal and non-terminal phij+1
        y = rewards + gamma * max_action_qvalues
        
        hn = torch.zeros(lstm_layers, lstm_n, dtype=torch.float32, device=device, requires_grad=True)
        cn = torch.zeros(lstm_layers, lstm_n, dtype=torch.float32, device=device, requires_grad=True)

        # optimizer.zero_grad()
        qvalues, (hn, cn) = self(states, (hn, cn))

        qvalues = qvalues.gather(1, actions)
        # loss = loss_fn(qvalues, y.unsqueeze(1))
        td_errors = torch.abs(qvalues - y.unsqueeze(1))
        td_errors = torch.clamp(td_errors, min=-1, max=1)
        loss = torch.square(td_errors) * weights.unsqueeze(1)

        # loss = loss.mean()
        # loss.backward()
        
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        # optimizer.step()
        
        replay_memory.update_priorities(indices, td_errors + 1e-5)
        return loss