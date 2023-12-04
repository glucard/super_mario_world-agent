from win32gui import FindWindow, GetWindowRect, SetForegroundWindow, GetForegroundWindow, GetWindowText
import mss.tools, ctypes
import win32api, win32process, win32con, win32com.client
import random, time, numpy as np

class GameCamera:
    def __init__(self, window_name, window_offset=(0,0,0,0) ):
        self.window_offset = window_offset
        self.shell = win32com.client.Dispatch("WScript.Shell")
        self.window_handle = FindWindow(None, window_name)
        self.window_rect = tuple(p + o for p, o in zip(GetWindowRect(self.window_handle), self.window_offset))

    def set_foreground_game(self):
        if GetForegroundWindow() != self.window_handle:
            self.shell.SendKeys('%')
            SetForegroundWindow(self.window_handle)

    def get_frame(self):
        with mss.mss() as sct:
            img = sct.grab(self.window_rect)
        return np.array(img)[:, :, :3]
    
class GameMemoryReader:
    def __init__(self, process_name):
        self.process_id, self.base_address = self.get_process_by_name(process_name)
        self.memory_pointers = {}
    
    def add_memory_pointer(self, pointer_name, address, offsets):
        self.memory_pointers[pointer_name] = {
            'address': address,
            'offsets': offsets
        }

    def get_value(self, pointer_name):
        pointer = self.memory_pointers[pointer_name]
        address = self.base_address+pointer['address']
        offsets = pointer['offsets']
        return self.read_process_memory(address, offsets)[1]

    def get_process_by_name(self, process_name):
        process_name = process_name.lower()
        processes = win32process.EnumProcesses()
        for process_id in processes:
            if process_id == -1:
                continue
            try:
                h_process = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ, True, process_id)
                try:
                    modules = win32process.EnumProcessModules(h_process)
                    for base_address in modules:
                        name = str(win32process.GetModuleFileNameEx(h_process, base_address))
                        if name.lower().find(process_name) != -1:
                            return process_id, base_address
                finally:
                    win32api.CloseHandle(h_process)
            except:
                pass
            
    def read_process_memory(self, address, offsets=[]):
        h_process = ctypes.windll.kernel32.OpenProcess(win32con.PROCESS_VM_READ, False, self.process_id)
        data = ctypes.c_uint(0)
        bytesRead = ctypes.c_uint(0)
        current_address = address
        if offsets:
            offsets.append(None)
            for offset in offsets:
                ctypes.windll.kernel32.ReadProcessMemory(h_process, current_address, ctypes.byref(data), ctypes.sizeof(data), ctypes.byref(bytesRead))
                if not offset:
                    return current_address, data.value
                else:
                    current_address = data.value + offset
        else:
            ctypes.windll.kernel32.ReadProcessMemory(h_process, current_address, ctypes.byref(data), ctypes.sizeof(data), ctypes.byref(bytesRead))
        ctypes.windll.kernel32.CloseHandle(h_process)
        return current_address, data.value

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
    }
    save_states = ['F1', 'F2', 'F3', 'F4']
    
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
            self.press_key('c')
        
        if action == 3:
            self.press_key('x')

        current_camera_pos = self.game_memory_reader.get_value('camera_pos')
        reward += 1 if current_camera_pos > self.last_camera_pos else -1
        # reward += -1 if current_camera_pos < self.last_camera_pos else 0
        self.last_camera_pos = current_camera_pos

        if self.mario_current_life_state != self.game_memory_reader.get_value('life_state') or current_camera_pos == 0:
            reward = -20
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