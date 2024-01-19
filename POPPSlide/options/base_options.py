import argparse

from options.read_config import load_config

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='Patient Level Prediction Model')

    def initialize(self):
        self.parser.add_argument('--config', '-c', type=str, default=None)

    def parse(self):
        args = self.parser.parse_args()
        print(args)
        if args.config:
            args = load_config(args)

            # basic check of argument validity
            if args.stratify is not None:
                assert(args.sampling_ratio is not None)
            if args.outcome_type == 'classification':
                assert(args.num_classes is not None)
            assert(args.train_level=='patient' or args.train_level=='slide')

            return args
        else:
            # basic check of argument validity
            if args.stratify is not None:
                assert(args.sampling_ratio is not None)
            return args
