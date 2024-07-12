from win32gui import FindWindow, GetWindowRect, SetForegroundWindow, GetForegroundWindow
import win32gui, win32con, win32ui
import mss.tools
import win32com.client
import numpy as np
from PIL import Image
from ctypes import windll

class GameCamera:
    def __init__(self, window_name, window_offset=(0,0,0,0)):
        self.window_offset = window_offset
        self.shell = win32com.client.Dispatch("WScript.Shell")
        self.window_handle = FindWindow(None, window_name)
        self.window_rect = GetWindowRect(self.window_handle)

    def set_foreground_game(self):
        if GetForegroundWindow() != self.window_handle:
            self.shell.SendKeys('%')
            SetForegroundWindow(self.window_handle)

    def old_get_frame(self):
        with mss.mss() as sct:
            img = sct.grab(self.window_rect)
        return np.array(img)[..., [2, 1, 0]]
    
    def get_frame(self):
        # get window dims
        left, top, right, bot = self.window_rect
        w = right - left
        h = bot - top

        # capture
        hwnd = self.window_handle
        wDC = win32gui.GetWindowDC(hwnd)
        dcObj=win32ui.CreateDCFromHandle(wDC)
        cDC=dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0,0),(w, h) , dcObj, (0,0), win32con.SRCCOPY)

        # Convert the bitmap to a Pillow image
        bmp_info = dataBitMap.GetInfo()
        bmp_str = dataBitMap.GetBitmapBits(True)
        img = Image.frombuffer('RGB', (bmp_info['bmWidth'], bmp_info['bmHeight']), bmp_str, 'raw', 'BGRX', 0, 1)

        # Free Resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())
        
        # resize img
        new_width, new_height = 180, 180 # Set your desired width and height
        resized_img = img.crop(self.window_offset).resize((new_width, new_height), Image.ANTIALIAS)

        # Convert the Pillow image to a NumPy array
        img_np = np.array(resized_img)

        return img_np