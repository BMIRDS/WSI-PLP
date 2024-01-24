import torchvision
import torch.nn as nn

# instantiate the model
def create_model(num_layers, pretrain, num_classes):
    assert num_layers in [18, 34, 50, 101, 152]
    architecture = "resnet{}".format(num_layers)
    model_constructor = getattr(torchvision.models, architecture)
    model = model_constructor(num_classes=num_classes)

    if pretrain is True:
        print("Loading pretrained model!")
        pretrained = model_constructor(pretrained=True).state_dict()
        if num_classes != pretrained['fc.weight'].size(0):
            del pretrained['fc.weight'], pretrained['fc.bias']
        model.load_state_dict(pretrained, strict=False)
    return model


class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class MobileNetV2Updated(nn.Module):

    def __init__(self, pretrained=False):
        super(MobileNetV2Updated, self).__init__()
        self.model = torchvision.models.mobilenet.mobilenet_v2(
            pretrained=pretrained)
        self.model.classifier = Identity()
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1280, 1000)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
