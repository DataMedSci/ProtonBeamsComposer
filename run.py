import argparse
import logging
from sys import argv

from pbc.optimize import basic_optimization


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-r', '--range', type=float, required=True)
    parser.add_argument('-s', '--spread', type=float, required=True)
    parser.add_argument('-f', '--full', type=str, choices=['range', 'spread', 'both'])
    parser.add_argument('-p', '--halfmod', action='store_true')
    parser.add_argument('-n', '--name', type=str, default='modulator')

    # advanced
    parser.add_argument('--smooth', action='store_true')
    parser.add_argument('--window', type=int, help="smooth window")
    parser.add_argument('-g', '--add_to_gott', type=int, help='if specified, add this number of peaks to calculated '
                                                              'with Gottschalk rule')
    parser.add_argument('-k', '--peaks', type=int, help='if specified, use this as number of peaks in optimization'
                                                        'and omit calculation using Gottschalk rule')

    # input files
    parser.add_argument('-i', '--input_bp_file', type=str)
    parser.add_argument('-d', '--delimiter', type=str, help='delimiter used in BP file')

    # logging
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-q', '--quiet', action='store_true')
    parser.add_argument('-o', '--no-plot', action='store_true')
    input_args = parser.parse_args(argv[1:])

    if input_args.verbose:
        log_level = logging.DEBUG
    elif input_args.quiet:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO

    stream_log = logging.StreamHandler()

    logging.basicConfig(format='--> (%(asctime)s) - %(levelname)s - [%(name)s.%(funcName)s:%(lineno)s]\n\t%(message)s',
                        datefmt='%H:%M:%S',
                        level=log_level,
                        handlers=[stream_log])

    basic_optimization(input_args)
