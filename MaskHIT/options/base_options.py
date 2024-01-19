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
            return args
        else:
            # basic check of argument validity
            return args
