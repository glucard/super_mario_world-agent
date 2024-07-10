from win32gui import FindWindow, GetWindowRect, SetForegroundWindow, GetForegroundWindow
import mss.tools
import win32com.client
import numpy as np
from PIL import Image

class GameCamera:
    def __init__(self, window_name, window_offset=(0,0,0,0)):
        self.window_offset = window_offset
        self.shell = win32com.client.Dispatch("WScript.Shell")
        self.window_handle = FindWindow(None, window_name)
        self.window_rect = tuple(p + o for p, o in zip(GetWindowRect(self.window_handle), self.window_offset))

    def set_foreground_game(self):
        if GetForegroundWindow() != self.window_handle:
            self.shell.SendKeys('%')
            SetForegroundWindow(self.window_handle)

    def old_get_frame(self):
        with mss.mss() as sct:
            img = sct.grab(self.window_rect)
        return np.array(img)[..., [2, 1, 0]]
    
    def get_frame(self):
        with mss.mss() as sct:
            screenshot = sct.grab(self.window_rect)
        # return np.array(screenshot)[..., [2, 1, 0]]

        # Convert the screenshot to a PIL Image
        img = Image.frombytes('RGB', (screenshot.width, screenshot.height), screenshot.rgb)

        # Resize the image
        new_width, new_height = 80, 80 # Set your desired width and height
        resized_img = img.resize((new_width, new_height)) #, Image.ANTIALIAS)

        return np.array(resized_img)