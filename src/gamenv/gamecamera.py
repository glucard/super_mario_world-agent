from win32gui import FindWindow, GetWindowRect, SetForegroundWindow, GetForegroundWindow, GetWindowText
import win32gui, win32con, win32ui
import win32com.client
import numpy as np
from PIL import Image
from ctypes import windll

from collections import deque
from threading import Thread, Lock
import time

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

    def check_window(self):
        """
        check if current game window is front window.
        """
        if GetForegroundWindow() != self.window_handle:
            raise ResourceWarning(f"{GetWindowText(self.window_handle)} cannot be minimized.")
        
    
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
        resized_img = img.convert("L").crop(self.window_offset).resize((new_width, new_height), Image.ANTIALIAS)

        # Convert the Pillow image to a NumPy array
        img_np = np.array(resized_img)

        return img_np[:,:,None]
    
class CameraFrameBuffer(GameCamera):
    def __init__(self, max_capacity: int, **kwarg):
        super(CameraFrameBuffer, self).__init__(**kwarg)
        self._max_capacity = max_capacity
        self.frame_buffer = None
        self.is_running = False
        self.thread = None
        self.lock = Lock()
        self.reset_buffer()

    def worker_t(self) -> None:
        n_times_per_second = 20  # The desired number of iterations per second
        interval = 1.0 / n_times_per_second  # Time interval between iterations in seconds
        max_capacity = self._max_capacity

        done = False
        while not done:
            start_time = time.time()
            try:
                self.check_window()
            except ResourceWarning as e:
                print(e)
                with self.lock:
                    self.is_running = False
                    return
                
            with self.lock:
                done = not self.is_running
                self.frame_buffer[:max_capacity-1] = self.frame_buffer[1:]
                self.frame_buffer[-1] = self.get_frame()
                
            elapsed_time = time.time() - start_time  # Calculate the time taken for the iteration
            sleep_time = max(0, interval - elapsed_time)  # Calculate the remaining time to sleep
            time.sleep(sleep_time)  # Sleep for the remaining time to maintain the desired frequency

    def start_buffer(self) -> None:
        with self.lock:

            if self.is_running:
                raise StopAsyncIteration("Buffer is already running")
            
            self.is_running = True
            self.thread = Thread(target=self.worker_t, daemon=True)
            self.thread.start()

    def stop_buffer(self) -> None:
        with self.lock:
            self.is_running = False
        
        self.thread.join()
        self.thread = None

    def get_frame_buffer(self) -> np.array:
        with self.lock:
            if not self.is_running:
                raise StopAsyncIteration("Frame buffer is not running")
            return self.frame_buffer.copy()

    def reset_buffer(self) -> None:
        frame = self.get_frame()
        shape = frame.shape
        frame_buffer = np.zeros((self._max_capacity, *shape), dtype=np.float32)

        for i in range(self._max_capacity):
            frame_buffer[i] = frame

        with self.lock:
            self.frame_buffer = frame_buffer


def debug():
    game_window_name="mario - Snes9x 1.62.3"
    window_offset=(10, 150, 550, 500)
    camera = CameraFrameBuffer(max_capacity=10, window_name=game_window_name, window_offset=window_offset)
    camera.set_foreground_game()
    camera.start_buffer()

    for i in range(10):
        print(camera.get_frame_buffer().shape)
        time.sleep(0.1)

    camera.stop_buffer()
    print(camera.is_running)

if __name__ == "__main__":
    debug()