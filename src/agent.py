import torch, torchvision

from .models.batch_model import DQN as BatchModel

CROP_DIMS = 60, 40, 95, 95
RESIZE = 150, 150
MODELS = {
    'batch_model': BatchModel,
}

class Agent:
    def __init__(self):
        pass

    def phi(observation, device):
        x = observation.transpose([2, 0, 1])
        x = torch.tensor(x, dtype=torch.float32, device=device)/255
        # x = torchvision.transforms.functional.crop(x, *CROP_DIMS)
        # x = torchvision.transforms.Grayscale()(x)
        x = torchvision.transforms.Resize(RESIZE, antialias=True)(x)
        # x = torchvision.transforms.Normalize(mean=x.mean(), std=x.std())(x)
        return x