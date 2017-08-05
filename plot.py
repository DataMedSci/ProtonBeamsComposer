import argparse
import logging
from sys import argv

from pbc.plotting import make_plots_from_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help="Path to data file that should be used in plotting")
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-q', '--quiet', action='store_true')
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

    make_plots_from_file(input_args.file)
