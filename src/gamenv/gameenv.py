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
        'PAUSE':0x13
    }
    save_states = ['F1', 'F2'] # , 'F3', 'F4']
    
    def __init__(self, executable_name, game_window_name, window_offset=(0,0,0,0)):

        # setting camera
        self.camera = GameCamera(game_window_name, window_offset)

        # setting memory reader
        self.game_memory_reader = GameMemoryReader(executable_name)
        self.game_memory_reader.add_memory_pointer('life', 0x0090F520, [0x90])
        self.game_memory_reader.add_memory_pointer('life_state', 0x0090F520, [0xB4])
        self.game_memory_reader.add_memory_pointer('camera_pos', 0x00012698, [0x18])

        # reset current state
        self.reset()
    
    def reset(self):

        save_state = random.choice(GameEnv.save_states)
        self.press_key(save_state)
        time.sleep(0.25)
        self.release_key(save_state)

        self.mario_current_life_state = self.game_memory_reader.get_value('life_state')
        self.last_camera_pos = self.game_memory_reader.get_value('camera_pos')

        return self.camera.get_frame() # obs

    def step(self, action):
        reward = 0
        game_over = False

        if GetForegroundWindow() != self.camera.window_handle:
            raise Exception(f"{GetWindowText(self.camera.window_handle)} cannot be minimized.") 

        #self.toggle_pause()
        #print("unpaused")

        if action == 0:
            self.release_key('left_arrow')
            self.press_key('right_arrow')
            self.release_key('x') # maybe action to unhold
            self.release_key('c')
        if action == 1:
            self.release_key('right_arrow')
            self.press_key('left_arrow')
            self.release_key('x')
            self.release_key('c')
        
        if action == 2:
            self.release_key('c')
            time.sleep(0.01)
            self.press_key('c')
        
        if action == 3:
            self.release_key('x')
            time.sleep(0.01)
            self.press_key('x')

        time.sleep(0.12) # wait
        #self.toggle_pause()
        #print("paused")

        current_camera_pos = self.game_memory_reader.get_value('camera_pos')
        reward += 1 if current_camera_pos > self.last_camera_pos else -1
        # reward += -1 if current_camera_pos < self.last_camera_pos else 0
        self.last_camera_pos = current_camera_pos

        if self.mario_current_life_state != self.game_memory_reader.get_value('life_state') or current_camera_pos == 0:
            self.release_key('left_arrow')
            self.release_key('right_arrow')
            self.release_key('c')
            self.release_key('x')
            reward = -50
            game_over = True
        # time.sleep(0.1)

        return [
            self.camera.get_frame(), # obs
            reward, # reward
            game_over, # game_over
        ]
    
    def press_key(self, key):
        win32api.keybd_event(GameEnv.VK_CODE[key], win32api.MapVirtualKey(GameEnv.VK_CODE[key], 0), 0, 0)

    def release_key(self, key):
        win32api.keybd_event(GameEnv.VK_CODE[key], win32api.MapVirtualKey(GameEnv.VK_CODE[key], 0), win32con.KEYEVENTF_KEYUP, 0)

    def toggle_pause(self):
        self.press_key('PAUSE')
        time.sleep(0.01)
        self.release_key('PAUSE')
        time.sleep(0.01)
