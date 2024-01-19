from .base_options import BaseOptions

"""Argument Parser for Training Parameters
"""

class TrainOptions(BaseOptions):
    def initialize(self):
        super().initialize()

        # specify model structure
        self.parser.add_argument('--backbone',
                            type=str, default='resnet-18',
                            help='backbone model, for example, resnet-50, mobilenet, etc')
        self.parser.add_argument('--class-weights',
                            type=str, default=None,
                            help='weights for each class, separated by comma')
        self.parser.add_argument('--outcome',
                            type=str, default='',
                            help='name of the outcome variable')
        self.parser.add_argument('--outcome-type',
                            type=str, default='survival',
                            help='outcome type, choose from "survival", "classification", "regression"')
        self.parser.add_argument('--num-classes',
                            type=int, default=2,
                            help='number of outputs of the model, only used for classification')
        # specify the path of the meta files
        self.parser.add_argument('--test-meta',
                            type=str, default='data/val-meta.pickle',
                            help='path to the meta file for the evaluation portion')
        self.parser.add_argument('--train-meta',
                            type=str, default='data/train-meta.pickle',
                            help='path to the meta file for the training portion')
        self.parser.add_argument('--patch-meta',
                            type=str, default='data/patch-meta.pickle',
                            help='path to the meta file for the training portion')

        # specify patch manipulations
        self.parser.add_argument('--crop-size',
                            type=int, default=224,
                            help='size of the crop')
        self.parser.add_argument('--num-crops',
                            type=int, default=1,
                            help='number of crops to extract from one patch')
        self.parser.add_argument('--num-patches',
                            type=int, default=8,
                            help='number of patches to select from one patient during one iteration')

        # learning rate
        self.parser.add_argument('--lr-backbone',
                            type=float, default=1e-5,
                            help='learning rate for the backbone model')
        self.parser.add_argument('--lr-head',
                            type=float, default=1e-5,
                            help='learning rate for the head model')
        self.parser.add_argument('--cosine-anneal-freq',
                            type=int, default=100,
                            help='anneal frequency for the cosine scheduler')
        self.parser.add_argument('--cosine-t-mult',
                            type=int, default=1,
                            help='t_mult for cosine scheduler')

        # specify experiment details
        self.parser.add_argument('-m', '--mode',
                            type=str, default='train',
                            help='mode, train or test')
        self.parser.add_argument('--patience',
                            type=int, default=100,
                            help='break the training after how number of epochs of no improvement')
        self.parser.add_argument('--epochs',
                            type=int, default=100,
                            help='total number of epochs to train the model')
        self.parser.add_argument('--pretrain',
                            action='store_true', default=False,
                            help='whether use a pretrained backbone')
        self.parser.add_argument('--random-seed',
                            type=int, default=1234,
                            help='random seed of the model')
        self.parser.add_argument('--resume',
                            type=str, default='',
                            help='path to the checkpoint file')

        # data specific arguments
        self.parser.add_argument('-b', '--batch-size',
                            type=int, default=8,
                            help='batch size')
        self.parser.add_argument('--stratify',
                            type=str, default=None,
                            help='whether to use a stratify approach when splitting the train/val/test datasets')
        self.parser.add_argument('--sampling-ratio',
                            type=str, default=None,
                            help='fixed sampling ratio for each class for each batch, for example 1,3')
        self.parser.add_argument('--repeats-per-epoch',
                            type=int, default=100,
                            help='how many times to select one patient during each iteration')
        self.parser.add_argument('--num-workers',
                            type=int, default=4,
                            help='number of CPU threads')
        self.parser.add_argument('--data-stats-mean',
                            type=float, nargs=3,
                            default=None,
                            help='patch mean of dataset')
        self.parser.add_argument('--data-stats-std',
                            type=float, nargs=3,
                            default=None,
                            help='patch std of dataset')

        # model regularization
        self.parser.add_argument('--dropout',
                            type=float, default=0,
                            help='dropout rate, not implemented yet')
        self.parser.add_argument('--wd-backbone',
                            type=float, default=0.0001,
                            help='intensity of the weight decay for the backbone model')
        self.parser.add_argument('--wd-head',
                            type=float, default=1e-5,
                            help='intensity of the weight decay for the head model')
        self.parser.add_argument('--l1',
                            type=float, default=0,
                            help='intensity of l1 regularization')
        self.parser.add_argument('--l2',
                            type=float, default=0,
                            help='intensity of l2 regularization')

        # evaluation details
        self.parser.add_argument('--sample-id',
                            action='store_true', default=False,
                            help='if true, sample patches by patient; otherwise evaluate the model on all patches')
        self.parser.add_argument('--num-val',
                            type=int, default=100,
                            help='number of patches to select from one patient during validation')

        # model monitoring
        self.parser.add_argument('--timestr',
                            type=str, default='',
                            help='time stamp of the model')
        self.parser.add_argument('--log-freq',
                            type=int, default=10,
                            help='how frequent (in steps) to print logging information')
        self.parser.add_argument('--save-interval',
                            type=int, default=1,
                            help='how frequent (in epochs) to save the model checkpoint')
        self.parser.add_argument('--checkpoint-dir',
                            type=str, default='../checkpoints',
                            help='path to model checkpoint directory')
        
