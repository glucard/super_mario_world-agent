from win32gui import GetForegroundWindow, GetWindowText
import win32api, win32con
import random, time

from .gamecamera import GameCamera
from .gamememoryreader import GameMemoryReader

import multiprocessing

import gymnasium
from gymnasium import spaces

import numpy as np

class GameEnv(gymnasium.Env):
    metadata = {'render.modes': ['human']}
    VK_CODE = {
        'c': 0x43,
        'left_arrow':0x25,
        'up_arrow':0x26,
        'right_arrow':0x27,
        'down_arrow':0x28,
        'x':0x58,
        'F1':0x70,
        'F2':0x71,
        'F3':0x72,
        'F4':0x73,
        'F5':0x74,
        'F6':0x75,
        'F7':0x76,
        'F8':0x77,
        'F9':0x78,
        'F10':0x79,
        ']':0xDD,
        'SPACE':0x20,
        'PAUSE':0x13,
        'p': 0x50,
    }
    save_states = ['F1'] #, 'F2'] # , 'F3', 'F4']
    checkpoint_distance = 1_000
    
    def __init__(self, executable_name="snes9x.exe", game_window_name="mario - Snes9x 1.62.3", window_offset=(20, 120, -10, -50)):
        super(GameEnv, self).__init__()

        # setting gym
        self.action_space = spaces.MultiDiscrete([3, 3])
        self.observation_space = spaces.Box(low=0, high=255, shape=(80, 80, 3), dtype=np.uint8)
        self.metadata = {'render.modes': ['human']}
        self._max_episode_steps = 200  # Set your maximum episode steps
        self._elapsed_steps = 0
        self.num_envs = 1

        # setting camera
        self.camera = GameCamera(game_window_name, window_offset)

        # setting memory reader
        self.game_memory_reader = GameMemoryReader(executable_name)
        self.game_memory_reader.add_memory_pointer('score', 0x0090193C, [0xC8])
        self.game_memory_reader.add_memory_pointer('life', 0x0090F520, [0x90])
        self.game_memory_reader.add_memory_pointer('life_state', 0x0090F520, [0xB4]) # its not life_state, change when player lost control of character
        self.game_memory_reader.add_memory_pointer('camera_pos', 0x00012698, [0x18])
        self.game_memory_reader.add_memory_pointer('change_on_level_end', 0x008EDE1C, [0x24])
        self.game_memory_reader.add_memory_pointer('change_on_going_back_to_map', 0x0002C9A0, [0x88]) # check death
        self.frame_buffer = []

        # reset current state
        self.reset()
    
    def reset(self, seed=None, options=None):
        self.camera.set_foreground_game()

        save_state = random.choice(GameEnv.save_states)
        self.press_key(save_state)
        time.sleep(0.5)
        self.release_key(save_state)

        self.last_reached_checkpoint = self.get_curr_checkpoint()
        self.frames_on_checkpoint_count = 0

        self.mario_current_life_state = self.game_memory_reader.get_value('change_on_going_back_to_map')
        #print(self.mario_current_life_state)
        self.last_camera_pos = self.game_memory_reader.get_value('camera_pos')
        self.current_end_state = self.game_memory_reader.get_value('change_on_level_end')
        self.score = self.game_memory_reader.get_value('score')
        self._elapsed_steps = 0

        obs = self.camera.get_frame()
        info = {}
        return obs, info
    
    def start_buffer():
        num_processes = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=num_processes)

    def step(self, action):
        reward = 0
        game_over = False

        if GetForegroundWindow() != self.camera.window_handle:
            raise Exception(f"{GetWindowText(self.camera.window_handle)} cannot be minimized.") 

        #self.toggle_pause()
        #print("unpaused")

        #self.press_key('right_arrow')
        
        #self.release_key('left_arrow')
        # self.release_key('right_arrow')
        # self.release_key('c')
        self.release_key('x')
        
        action1, action2 = action

        if action1 == 0:
            self.release_key('left_arrow')
            self.press_key('right_arrow')
            # self.release_key('x') # maybe action to unhold
            # self.release_key('c')
            
        elif action1 == 2:
            self.release_key('right_arrow')
            self.release_key('left_arrow')
            # self.release_key('x')
            # self.release_key('c')
            
        elif action1 == 1:
            self.release_key('right_arrow')
            self.press_key('left_arrow')
            # self.release_key('x')
            # self.release_key('c')

        if action2 == 0:
            # reward += -0.5
            self.press_key('c')
        
        elif action2 == 1:
            self.release_key('x')
            self.release_key('c')

        elif action2 == 2:
            self.release_key('x')
            self.press_key('x')
            
        # self.skip_frame(10)

        time.sleep(0.01) # wait
        #self.toggle_pause()
        #print("paused")
        #self.release_key('c')
        """
        curr_score = self.game_memory_reader.get_value('score')
        if self.score < curr_score:
            print("mario scored")
            self.score = curr_score
            reward += 5
        """
        current_camera_pos = self.game_memory_reader.get_value('camera_pos')

        curr_checkpoint = self.get_curr_checkpoint()
        if curr_checkpoint > self.last_reached_checkpoint:
            #print("mario advanced")
            # if moving right
            self.last_reached_checkpoint = curr_checkpoint
            reward += 0.5
            self.frames_on_checkpoint_count = 0
        elif curr_checkpoint < self.last_reached_checkpoint:
            #print("mario backed")
            # if moving left 
            self.last_reached_checkpoint = curr_checkpoint
            reward += -0.5
            self.frames_on_checkpoint_count = 0
        else:
            #print("mario standing")
            # if not moving
            reward += 0
        self.frames_on_checkpoint_count += 1

        if self.frames_on_checkpoint_count > 50:
            #print("mario is stuck")
            reward += -1
            game_over = True
        # reward += 1 if current_camera_pos > self.last_camera_pos else -1
        # reward += -1 if current_camera_pos < self.last_camera_pos else 0
        self.last_camera_pos = current_camera_pos

        if current_camera_pos == 0:
            # if back to map start
            reward += -1
            game_over = True

        if self.mario_current_life_state != self.game_memory_reader.get_value('change_on_going_back_to_map'): # or current_camera_pos == 0:
            # if died
            #print("mario died")
            reward += -2
            game_over = True
            
        if self.current_end_state != self.game_memory_reader.get_value('change_on_level_end'):
            # if win
            #print("mario win")
            reward += 100
            game_over = True
            #print(reward)
        
        if game_over:
            self.release_key('left_arrow')
            self.release_key('right_arrow')
            self.release_key('c')
            self.release_key('x')
            
        self._elapsed_steps += 1
        done = game_over
        truncated = self._elapsed_steps >= self._max_episode_steps
        info = {}
        return self.camera.get_frame(), reward, done, truncated, info
    
    def render(self, mode='human'):
        if mode == 'human':
            pass

    def close(self):
        # Clean up resources (optional)
        pass
    def get_curr_checkpoint(self):

        current_camera_pos = self.game_memory_reader.get_value('camera_pos')
        return current_camera_pos // GameEnv.checkpoint_distance

    def press_key(self, key):
        win32api.keybd_event(GameEnv.VK_CODE[key], win32api.MapVirtualKey(GameEnv.VK_CODE[key], 0), 0, 0)

    def release_key(self, key):
        win32api.keybd_event(GameEnv.VK_CODE[key], win32api.MapVirtualKey(GameEnv.VK_CODE[key], 0), win32con.KEYEVENTF_KEYUP, 0)

    def toggle_pause(self):
        self.press_key('PAUSE')
        time.sleep(0.2)
        self.release_key('PAUSE')
        time.sleep(0.2)
    
    def skip_frame(self, n_frames):
        for _ in range(n_frames):
            self.press_key('p')
            #self.release_key('p')
            #time.sleep(0.1)

