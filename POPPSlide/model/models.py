import torch
import torch.nn as nn
import torch.nn.functional as F
from .helper import create_model, MobileNetV2Updated

########################################
# the overall model
########################################

class HybridModel(nn.Module):
    def __init__(
            self,
            backbone='resnet-18',
            pretrain=True,
            outcome_dim=1,
            outcome_type='survival',
            feature_dim=512,
            random_seed=1,
            dropout=0.1,
            output_features=False,
            device=torch.device("cpu")
            ):
        super(HybridModel, self).__init__()
        self.output_features = output_features
        self.device = device

        torch.random.manual_seed(random_seed)

        # initialize model
        if isinstance(backbone, str):
            backbone_arch = backbone.split('-')[0]
            if backbone_arch == 'resnet':
                backbone_nlayers = int(backbone.split('-')[1])
                self.backbone = create_model(backbone_nlayers, pretrain, 1)
            elif backbone_arch == 'mobilenet':
                self.backbone = MobileNetV2Updated(pretrain)
                backbone_nlayers = 0

            else:
                print("The backbone %s is not supported!" % backbone_arch)

            model_dim = {
                'resnet': {
                    18: 512,
                    34: 512,
                    50: 2048
                },
                'mobilenet': {
                    0: 1280
                }
            }
            self.feature_dim = model_dim[backbone_arch][backbone_nlayers]

        else:
            self.backbone = backbone
            self.feature_dim = feature_dim

        self.backbone.fc = nn.Identity()
        self.backbone = self.backbone.to(self.device)  # same as model.cuda()

        # if multiple GPUs
        if torch.cuda.device_count() > 1:
            self.backbone = nn.DataParallel(self.backbone)

        self.head = MLP(
                in_dim=self.feature_dim,
                out_dim=outcome_dim,
                outcome_type=outcome_type,
                dropout=dropout
            )
        self.head = self.head.to(self.device)  # same as model.cuda()

    def forward(self, x, ppi):
        features = self.backbone(x)
        out = self.head(features.view(x.size(0) // ppi, ppi, -1))
        if self.output_features:
            out['features'] = features
        return out


########################################
# MLP models
# average the patch features
# make predictions
########################################


class MLP(nn.Module):
    def __init__(
            self,
            in_dim=512,
            out_dim=1,
            outcome_type='survival',
            dropout=0.1):

        super(MLP, self).__init__()
        if outcome_type == 'survival':
            self.fc = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(128, 1, bias=False),
                nn.Tanh()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(128, out_dim, bias=True)
            )

    def forward(self, x):
        x = x.mean(dim=1)
        out = {
            'pred': self.fc(x)
        }
        return out
