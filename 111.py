import torch
print(torch.__version__)              # → 應該會看到 2.7.1+cu118
print(torch.cuda.is_available())      # → 若成功，會顯示 True
print(torch.cuda.get_device_name(0))
