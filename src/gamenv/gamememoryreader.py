import ctypes
import win32api, win32process, win32con
    
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