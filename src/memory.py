import random, torch
from collections import deque
from itertools import islice
import numpy as np
import threading, time

def synchronized_method(func):
    def synchronized_func(self, *args, **kwargs):
        with self.memory_lock:
            return func(self, *args, **kwargs)
    return synchronized_func

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
    def render_simple(self) -> torch.tensor:
        x = torch.stack(tuple(self.memory))
        return x
    
    def clear(self):
        self.memory.clear()

class PrioritizedReplayMemory:
    
    def __init__(self, capacity, torch_device, prob_alpha=0.6, queue_push_thread = False):

        def que_push_thread_func(self):
            while True:
                # print("que_push_thread_func: pushing queue into memory...")
                with self.memory_lock:
                    # with self.push_memory_lock:
                        while len(self.push_queue_memory) > 0:
                            new_memory = self.push_queue_memory.pop(0)
                            self.push(*new_memory)
                time.sleep(1)

        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.prob_alpha = prob_alpha
        self.torch_device = torch_device
        self.memory_lock = threading.Lock()
        self.push_memory_lock = threading.Lock()
        self.push_queue_memory = []

        if queue_push_thread:
            print("queue thread enabled")
            self.thread = threading.Thread(target=que_push_thread_func, args=[self], daemon=True)

    def start_queue_thread(self):
        if hasattr(self, "thread"):
            if self.thread.isDaemon():
                self.thread.start()

    def __len__(self):
        return len(self.memory)
    
    def max_priority(self):
        states, actions, rewards, next_states, priorities = *zip(*self.memory), # let the ',' to not give syntax error
        return max(priorities)

    def push(self, state, action, reward, next_state):
        max_prio  = self.max_priority() if len(self) > 0 else 1.0
        experience = (state, action, reward, next_state, max_prio)
        self.memory.append(experience)

    def push_queue(self, state, action, reward, next_state):
        experience = (state, action, reward, next_state)
        # with self.memory_lock:
        self.push_queue_memory.append(experience)
    
    def update_priorities(self, indices, new_priorities):
        
        states, actions, rewards, next_states, priorities = *zip(*self.memory), # let the ',' to not give syntax error

        for idx, prio in zip(indices, new_priorities):
            prio = prio.item()
            self.memory[idx]= states[idx], actions[idx], rewards[idx], next_states[idx], prio

    def sample_sequence(self, batch_size, beta=0.4): # maybe separate by episode to avoid sequence where final_state -> start_state
        states, actions, rewards, next_states, priorities = *zip(*self.memory), # let the ',' to not give syntax error
        priorities = np.array(priorities)
        probs = priorities ** self.prob_alpha
        probs /= sum(probs)

        m = len(self)

        choice_indice = np.random.choice(m, p=probs) + 1
        choice_indice = choice_indice if choice_indice > batch_size else batch_size
        min_range = choice_indice - (batch_size)
        indices = list(range(min_range, choice_indice, 1))

        samples = [self.memory[idx] for idx in indices]
        
        weights = (m * probs[indices] ** (-beta))
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32).to(self.torch_device)

        states, actions, rewards, next_states, priorities = *zip(*samples),
        return states, actions, rewards, next_states, priorities, indices, weights
    
    # for sequence
    def render(self):
        if len(self.memory) < self.capacity:
            return False, None
        x = torch.stack(tuple(self.memory))
        return True, x
    
    def render_simple(self):
        x = torch.stack(tuple(self.memory))
        return x
    
    def clear(self):
        self.memory.clear()