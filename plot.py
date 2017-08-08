import argparse
import logging
from sys import argv

from pbc.plotting import make_plots_from_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help="Path to data file that should be used in plotting")
    parser.add_argument('-d', '--delimiter', type=str, help="Delimiter used in file")

    parser.add_argument('-c', '--second_file', help="Path to second file with data used to compare plots")
    parser.add_argument('-b', '--sec_delimiter', help="Delimiter used in second file")

    parser.add_argument('-p', '--plottype', type=str, choices=['sobp', 'plateau', 'none'], default='none')
    parser.add_argument('-s', '--savepath', type=str, help="If specified, used as path for saving the plot")

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

    make_plots_from_file(file_path=input_args.file,
                         delimiter=input_args.delimiter,
                         plottype=input_args.plottype,
                         save_path=input_args.savepath,
                         second_file=input_args.second_file,
                         second_file_delimiter=input_args.sec_delimiter)
