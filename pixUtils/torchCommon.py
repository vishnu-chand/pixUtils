import torch
from pixUtils import *
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"


def img2torch(x, device):
    if type(x) == list:
        x = np.array(x)
    x = torch.from_numpy(x)
    x = x.permute(2, 0, 1) if len(x.shape) == 3 else x.permute(0, 3, 1, 2)
    return x.to(device)


def mask2torch(x, device):
    if type(x) == list:
        x = np.array(x)
    x = torch.from_numpy(x)
    x = torch.unsqueeze(x, dim=1) if len(x.shape) == 3 else x[None]
    return x.to(device)


def torch2img(x):
    x = x.permute(1, 2, 0) if len(x.shape) == 3 else x.permute(0, 2, 3, 1)
    return x.cpu().numpy()


def torch2mask(x):
    x = torch.squeeze(x, dim=1) if len(x.shape) == 3 else x[0]
    return x.cpu().numpy()
