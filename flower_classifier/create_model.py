from torch import nn
from torchvision import models


def create_model(arch: str, n_hidden: int):
    model, n_input = None, None
    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        n_input = 9216
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        n_input = 25088
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        n_input = 1024
    else:
        print('Wrong architecture, only "alexnet", "vgg13", "densenet121" are available.')
        exit()

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(n_input, n_hidden),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(n_hidden, 102),
                                     nn.LogSoftmax(dim=1))
    return model
