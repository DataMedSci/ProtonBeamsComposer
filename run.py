import argparse
import logging
from sys import argv

from pbc.optimize import basic_optimization


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--range', type=float, required=True)
    parser.add_argument('-s', '--spread', type=float, required=True)
    parser.add_argument('-f', '--full', type=str, choices=['range', 'spread', 'both'])
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-q', '--quiet', action='store_true')
    parser.add_argument('-p', '--halfmod', action='store_true')
    input_args = parser.parse_args(argv[1:])

    if input_args.verbose:
        log_level = logging.DEBUG
    elif input_args.quiet:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO

    logging.basicConfig(format='--> (%(asctime)s)[%(name)s] - %(levelname)s - %(message)s',
                        datefmt='%H:%M:%S',
                        level=log_level)

    basic_optimization(input_args)
