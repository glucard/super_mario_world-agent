import random, torch
from collections import deque
from itertools import islice

class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
    
    def __len__(self):
        return len(self.memory)
    
    def push(self, experience):
        self.memory.append(experience)
    
    # for replay memory
    def sample(self, batch_size): # maybe separate by episode to avoid sequence where final_state -> start_state
        return list(random.sample(self.memory, batch_size))
    
    
    def sample_sequence(self, batch_size): # maybe separate by episode to avoid sequence where final_state -> start_state
        rand_range = random.randint(batch_size, len(self.memory))
        return list(islice(self.memory, rand_range - batch_size, rand_range))
    
    # for sequence
        
    def render(self):
        if len(self.memory) < self.capacity:
            return False, None
        x = torch.stack(tuple(self.memory))
        return True, x
    
    def clear(self):
        self.memory.clear()