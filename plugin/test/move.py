import torch
import dlwrapper
import random


class testmodel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # implement a simple 5 layer mlp
        self.conv = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


model = testmodel()
torch.save(model.state_dict(), "testmodel.th")
ckpt = dlwrapper.ModelManagerTorch("testmodel.th")
dat = ckpt.load()
print(dat)

ans_ckpt = torch.load("testmodel.th", map_location="cuda")

for k1, v1 in ans_ckpt.items():
    v2 = dat[k1]
    assert list(v1.shape) == v2.shape
    if v1.numel() <= 1024:
        continue
    v2.segment = [4, 4]
    x = random.randint(0, 3)
    y = random.randint(0, 3)
    grid = v2.create_tensor_grid()
    slice = grid.get_slice(x * 4 + y)
    assert slice.move_to(dlwrapper.MemoryType.PINNED)
    slice.wait()
    assert slice.move_to(dlwrapper.MemoryType.DEVICE)
    slice.wait()
    a, b = tuple(v1.shape)
    a //= 4
    b //= 4
    t1 = v1[x * a : (x + 1) * a, y * b : (y + 1) * b]
    t2 = slice.torch_get_contiguous(dlwrapper.MemoryType.DEVICE)
    print(t1 == t2)
