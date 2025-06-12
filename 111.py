import os
import ctypes

cuda_bin_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin"
os.add_dll_directory(cuda_bin_path)

print("👉 檢查 DLL 載入狀態...\n")

try:
    ctypes.WinDLL("cudart64_120.dll")
    print("✅ 成功載入 cudart64_120.dll")
except Exception as e:
    print("❌ cudart64_120.dll 載入失敗：", e)

try:
    ctypes.WinDLL("cudnn64_8.dll")
    print("✅ 成功載入 cudnn64_8.dll")
except Exception as e:
    print("❌ cudnn64_8.dll 載入失敗：", e)