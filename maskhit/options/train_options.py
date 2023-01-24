from .base_options import BaseOptions
"""Argument Parser for Training Parameters
"""


class TrainOptions(BaseOptions):

    def initialize(self):
        super().initialize()
        
        # model structure
        self.parser.add_argument('--mil1',
                                 type=str,
                                 default='vit_h8l12',
                                 help='attention method')
        self.parser.add_argument('--mil2',
                                 type=str,
                                 default='ap',
                                 help='fuse method')
        self.parser.add_argument('--num-features',
                                 type=int,
                                 default=512,
                                 help='dimension of patch features')
        self.parser.add_argument('--dropout',
                                 type=float,
                                 default=0,
                                 help='dropout rate')

        # experiments key options
        self.parser.add_argument('-m',
                                 '--mode',
                                 type=str,
                                 default='train',
                                 help='mode, train or predict or test')
        self.parser.add_argument('--test-type', type=str, default='val')
        self.parser.add_argument('--study',
                                 type=str,
                                 default=None,
                                 help="log files will be saved to logs/<study>/")
        self.parser.add_argument('--timestr',
                                 type=str,
                                 default='',
                                 help='time stamp of the model')
        self.parser.add_argument('--preset',
                                 type=str,
                                 default='',
                                 help='NOT IMPLEMENTED: pre-set mode')

        # specify dataset to use
        self.parser.add_argument('--meta-svs',
                                 type=str,
                                 default=None,
                                 help='meta file for svs files')
        self.parser.add_argument('--meta-train',
                                 type=str,
                                 default=None,
                                 help='meta file for train split')
        self.parser.add_argument('--meta-val',
                                 type=str,
                                 default=None,
                                 help='meta file for val split')
        self.parser.add_argument('--meta-all',
                                 type=str,
                                 default=None,
                                 help='meta contains both train and validation, override meta_train and meta_val')
        self.parser.add_argument('--cancer',
                                 type=str,
                                 default='',
                                 help='select cancer subset, if empty then use entire dataset')
        self.parser.add_argument('--fold',
                                 type=int,
                                 default=0,
                                 help='cross validation fold')
        self.parser.add_argument('--magnification',
                                 type=int,
                                 default=10,
                                 help='magnification')
        self.parser.add_argument('--patch-size',
                                 type=int,
                                 default=224,
                                 help='size of patch')
        self.parser.add_argument('--use-features',
                                 action='store_true',
                                 default=False,
                                 help='use extracted features')
        self.parser.add_argument('--ffpe-only',
                                 action='store_true',
                                 default=False,
                                 help='only ffpe slides')
        self.parser.add_argument('--ffpe-exclude',
                                 action='store_true',
                                 default=False,
                                 help='exclude ffpe slides')
        self.parser.add_argument('--num-svs',
                                 type=int,
                                 default=1,
                                 help='number of svs sampled in sample-patient mode')

        # outcome infomation
        self.parser.add_argument('--outcome',
                                 type=str,
                                 default='status',
                                 help='name of the outcome variable')
        self.parser.add_argument('--outcome-type',
                                 type=str,
                                 default='survival',
                                 help='outcome type, choose from \
                                "survival", "classification", "regression", "mlm"')
        self.parser.add_argument('--weighted-loss',
                                 action='store_true',
                                 default=False,
                                 help="HASN'T TESTED: whether to use a weighted loss function for imbalanced classification")

        # sample selection
        self.parser.add_argument('--tile-size',
                                 type=int,
                                 default=4480,
                                 help='size of tile region sampled from svs, if none then sample from entire svs')
        self.parser.add_argument('--num-patches',
                                 type=int,
                                 default=400,
                                 help='number of patches to select from one patient during one iteration')
        self.parser.add_argument('--sample-all',
                                 action='store_true',
                                 default=False,
                                 help='sample all the patches from \
                                whole slide image sequentially')
        self.parser.add_argument('--sampling-threshold',
                                 type=int,
                                 default=100,
                                 help='threshold when selecting tiles')
        self.parser.add_argument('--num-patches-val',
                                 type=int,
                                 default=0,
                                 help=
                                 'number of patches to select from one patient during one iteration at validation time')
        self.parser.add_argument('--repeats-per-svs',
                                 type=int,
                                 default=4,
                                 help='how many times to select one svs during each iteration')
        self.parser.add_argument('--repeats-per-epoch',
                                 type=int,
                                 default=4,
                                 help='how many times to select one patient during each epoch')
        self.parser.add_argument('--sample-patient',
                                 action='store_true',
                                 default=False,
                                 help='sample by patient')
        self.parser.add_argument('--sample-svs',
                                 action='store_true',
                                 default=False,
                                 help='sample by svs')

        # vit model specific parameters
        self.parser.add_argument('--avg-cls',
                                 action='store_true',
                                 default=False,
                                 help='average the sequence encoding to obtain cls')
        self.parser.add_argument('--hidden-dim',
                                 type=int,
                                 default=512,
                                 help='hidden dim for transformer')
        self.parser.add_argument('--zero-padding',
                                 action='store_true',
                                 default=False,
                                 help='use stored zero paddings')
        self.parser.add_argument('--mlp-dim',
                                 type=int,
                                 default=2048,
                                 help='mlp dim')
        self.parser.add_argument('--disable-position',
                                 action='store_true',
                                 default=False,
                                 help='disable postional encoding')

        # deepattnmisl specific parameters
        self.parser.add_argument('--cluster-centers',
                                 type=str,
                                 default='',
                                 help='location to cluster centers')

        # training schedule
        self.parser.add_argument('-b',
                                 '--batch-size',
                                 type=int,
                                 default=8,
                                 help='batch size')
        self.parser.add_argument('--epochs',
                                 type=int,
                                 default=100,
                                 help='total number of epochs to train the model')
        self.parser.add_argument('--warmup-epochs',
                                 type=int,
                                 default=0,
                                 help='total number of epochs to train the model')
        self.parser.add_argument('--patience',
                                 type=int,
                                 default=5,
                                 help='break the training after how number of epochs of no improvement')
        self.parser.add_argument('--anneal-freq',
                                 type=int,
                                 default=50,
                                 help='CosineAnnealingWarmRestarts: anneal frequency')
        self.parser.add_argument('--t-mult',
                                 type=int,
                                 default=2,
                                 help='CosineAnnealingWarmRestarts: A factor increases Ti after a restart')

        # specify optimizer
        self.parser.add_argument('--optimizer',
                                 type=str,
                                 default='adamw',
                                 help='optimizer, choose from "adam","adamw",""')
        self.parser.add_argument('--lr-attn',
                                 type=float,
                                 default=1e-5,
                                 help='learning rate for the attn part')
        self.parser.add_argument('--lr-pred',
                                 type=float,
                                 default=1e-5,
                                 help='learning rate for the pred part')
        self.parser.add_argument('--lr-loss',
                                 type=float,
                                 default=1e-5,
                                 help='learning rate for the mlm loss part')
        self.parser.add_argument('--lr-fuse',
                                 type=float,
                                 default=1e-5,
                                 help='learning rate for the fuse part')
        self.parser.add_argument('--lr',
                                 type=float,
                                 default=None,
                                 help='learning rate, if not none then will override all other learning rates')

        # weight decay
        self.parser.add_argument('--wd-attn',
                                 type=float,
                                 default=0.0001,
                                 help='intensity of the weight decay for the attn model')
        self.parser.add_argument('--wd-pred',
                                 type=float,
                                 default=1e-5,
                                 help='intensity of the weight decay for the pred model')
        self.parser.add_argument('--wd-loss',
                                 type=float,
                                 default=0.0001,
                                 help='intensity of the weight decay for the attn model')
        self.parser.add_argument('--wd-fuse',
                                 type=float,
                                 default=1e-5,
                                 help='intensity of the weight decay for the pred model')
        self.parser.add_argument('--wd',
                                 type=float,
                                 default=None,
                                 help='intensity of the weight decay')

        # repeated test for stable result estimation
        self.parser.add_argument('--num-repeats',
                                 type=int,
                                 default=3,
                                 help='number of times of repeats during test')
        self.parser.add_argument('--num-val',
                                 type=int,
                                 default=5,
                                 help='number of svs files to select from one patient during validation')

        # performance config
        self.parser.add_argument('-j',
                                 '--num-workers',
                                 type=int,
                                 default=10,
                                 help='number of CPU threads')
        self.parser.add_argument('--gpu',
                                 default=None,
                                 type=int,
                                 help='GPU id to use.')

        # resume from checkpoint
        self.parser.add_argument('--resume',
                                 type=str,
                                 default='',
                                 help='name of the model to be resumed')
        self.parser.add_argument('--resume-epoch',
                                 type=str,
                                 default='LAST',
                                 help='epoch of the model to be resumed')
        self.parser.add_argument('--resume-train',
                                 action='store_true',
                                 default=False,
                                 help='If true then continue training the \
                                resumed model using the same model name. \
                                If false, training using a new model name')
        self.parser.add_argument('--resume-optim',
                                 action='store_true',
                                 default=False,
                                 help='If true then resume the previous optim scheduler')
        self.parser.add_argument('--resume-fuzzy',
                                 action='store_true',
                                 default=False,
                                 help='turn off strict mode')

        # patch region masking
        self.parser.add_argument('--prob-mask',
                                 type=float,
                                 default=0,
                                 help='mask probability in BERT')
        self.parser.add_argument('--prop-mask',
                                 type=str,
                                 default='0,1,0',
                                 help='proportion for masking tokens, masked:original:random')
        self.parser.add_argument('--block-min',
                                 type=int,
                                 default=1,
                                 help='min size when masking blocks')
        self.parser.add_argument('--block-max',
                                 type=int,
                                 default=1,
                                 help='max size when masking blocks')

        # pretraining options
        self.parser.add_argument('--num-clusters',
                                 type=int,
                                 default=128,
                                 help='mlm: number of clusters to predict')
        self.parser.add_argument('--mlm-loss',
                                 type=str,
                                 default='null',
                                 help='cluster: only for sequence '
                                 'infonce: compare with memory '
                                 'infonce2: compare with self '
                                 'crossentropy: only for sequence')
        self.parser.add_argument('--no-cls-loss',
                                 action='store_true',
                                 default=False,
                                 help='whether to disable cls token loss')
        self.parser.add_argument('--no-seq-loss',
                                 action='store_true',
                                 default=False,
                                 help='whether to disable seq token loss')

        # experiment optional options
        self.parser.add_argument('--checkpoints-folder',
                                 type=str,
                                 default='checkpoints',
                                 help='path to the checkpoints folder')
        self.parser.add_argument('--log-freq',
                                 type=int,
                                 default=10,
                                 help='how frequent (in steps) to print logging information')
        self.parser.add_argument('--save-interval',
                                 type=int,
                                 default=100,
                                 help='how frequent (in epochs) to save the model checkpoint')
        self.parser.add_argument('--not-save',
                                 action='store_true',
                                 default=False,
                                 help='not save checkpoint')
        self.parser.add_argument('--override-logs',
                                 action='store_true',
                                 default=False,
                                 help='override logs')
        self.parser.add_argument('--notes',
                                 type=str,
                                 default='',
                                 help='custom note of this experiment')


        # feature extraction for visualization
        self.parser.add_argument('--visualization',
                                 action='store_true',
                                 default=False,
                                 help='vis')
        self.parser.add_argument('--vis-layer',
                                 type=int,
                                 default=None)
        self.parser.add_argument('--vis-head',
                                 type=int,
                                 default=None)