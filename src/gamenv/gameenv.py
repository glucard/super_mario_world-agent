from win32gui import GetForegroundWindow, GetWindowText
import win32api, win32con
import random, time

from .gamecamera import GameCamera
from .gamememoryreader import GameMemoryReader

class GameEnv:

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
    save_states = ['F1'] #, 'F2', 'F3', 'F4']
    checkpoint_distance = 1_000
    
    def __init__(self, executable_name, game_window_name, window_offset=(0,0,0,0)):

        # setting camera
        self.camera = GameCamera(game_window_name, window_offset)

        # setting memory reader
        self.game_memory_reader = GameMemoryReader(executable_name)
        self.game_memory_reader.add_memory_pointer('life', 0x0090F520, [0x90])
        self.game_memory_reader.add_memory_pointer('life_state', 0x0090F520, [0xB4]) # its not life_state, change when player lost control of character
        self.game_memory_reader.add_memory_pointer('camera_pos', 0x00012698, [0x18])
        self.game_memory_reader.add_memory_pointer('change_on_level_end', 0x008EDE1C, [0x24])
        self.game_memory_reader.add_memory_pointer('change_on_going_back_to_map', 0x0002C9A0, [0x88]) # check death

        # reset current state
        self.reset()
    
    def reset(self):

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

        return self.camera.get_frame() # obs

    def step(self, action):
        reward = 0
        game_over = False

        if GetForegroundWindow() != self.camera.window_handle:
            raise Exception(f"{GetWindowText(self.camera.window_handle)} cannot be minimized.") 

        #self.toggle_pause()
        #print("unpaused")

        self.release_key('left_arrow')
        self.release_key('right_arrow')
        # self.release_key('c')
        self.release_key('x')

        if action == 0:
            # self.release_key('left_arrow')
            self.press_key('right_arrow')
            # self.release_key('x') # maybe action to unhold
            # self.release_key('c')

        elif action == 1:
            reward += -0.1
            self.release_key('c')
            time.sleep(0.01)
            self.press_key('c')

        elif action == 2:
            # self.release_key('right_arrow')
            self.press_key('left_arrow')
            # self.release_key('x')
            # self.release_key('c')
        
        
        elif action == 3:
            self.release_key('x')
            time.sleep(0.01)
            self.press_key('x')
            
        # self.skip_frame(10)

        time.sleep(0.04) # wait
        #self.toggle_pause()
        #print("paused")
        #self.release_key('c')
        
        
        current_camera_pos = self.game_memory_reader.get_value('camera_pos')

        curr_checkpoint = self.get_curr_checkpoint()
        if curr_checkpoint > self.last_reached_checkpoint:
            # if moving right
            self.last_reached_checkpoint = curr_checkpoint
            reward += 0.5
            self.frames_on_checkpoint_count = 0
        elif curr_checkpoint < self.last_reached_checkpoint:
            # if moving left 
            self.last_reached_checkpoint = curr_checkpoint
            reward += -0.1
            self.frames_on_checkpoint_count = 0
        else:
            # if not moving
            reward += 0

        self.frames_on_checkpoint_count += 1

        if self.frames_on_checkpoint_count > 100:
            reward += -5
            game_over = True
        # reward += 1 if current_camera_pos > self.last_camera_pos else -1
        # reward += -1 if current_camera_pos < self.last_camera_pos else 0
        #self.last_camera_pos = current_camera_pos
        
        if current_camera_pos == 0:
            # if back to map start
            reward += -1
            game_over = True

        if self.mario_current_life_state != self.game_memory_reader.get_value('change_on_going_back_to_map'): # or current_camera_pos == 0:
            # if died
            reward += -10
            game_over = True
            
        if self.current_end_state != self.game_memory_reader.get_value('change_on_level_end'):
            # if win
            reward += 100
            game_over = True
            print(reward)

        
        if game_over:
            self.release_key('left_arrow')
            self.release_key('right_arrow')
            self.release_key('c')
            self.release_key('x')

        return [
            self.camera.get_frame(), # obs
            reward, # reward
            game_over, # game_over
        ]
    
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
