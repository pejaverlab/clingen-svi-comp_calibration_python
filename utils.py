import os
import csv
import argparse
import numpy as np

def load_labelled_data(filepath):
    data = None
    with open(filepath, "r") as f:
        reader = csv.reader(f, delimiter='\t')
        data = list(reader)
        f.close()
    x = np.array([float(e[0]) for e in data])
    y = np.array([int(e[1]) for e in data])
    return x,y

def load_unlabelled_data(filepath):
    data = None
    with open(filepath, "r") as f:
        reader = csv.reader(f, delimiter='\t')
        data = list(reader)
        f.close()
    g = np.array([float(e[0]) for e in data])
    return g


def getParser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    parser.add_argument(
        "infer",
        action='store_true',
    )
    parser.add_argument(
        "calibrate",
        action='store_true',
    )

    parser_calibrate = subparsers.add_parser('calibrate')
    parser_infer = subparsers.add_parser('infer')

    parser_calibrate.add_argument(
        "--configfile",
        default=None,
        type=str,
        required=True,
    )

    parser_calibrate.add_argument(
        "--outdir",
        default="out",
        type=str,
        required=True,
    )
    parser_calibrate.add_argument(
        "--labelled_data_file",
        default=None,
        type=str,
        required=True,
    )
    parser_calibrate.add_argument(
        "--unlabelled_data_file",
        default=None,
        type=str,
        required=False,
    )
    parser_calibrate.add_argument(
        "--reverse",
        action='store_true',
    )

    parser_infer.add_argument(
        "--score",
        default=None,
        type = float,
        required=False,
    )

    parser_infer.add_argument(
        "--score_file",
        default=None,
        type=str,
        required=False,
    )

    parser_infer.add_argument(
        "--calibrated_data_directory",
        default=None,
        type=str,
        required=False
    )
    
    parser_infer.add_argument(
        "--tool_name",
        default=None,
        type=str,
        required=False
    )

    parser_infer.add_argument(
        "--reverse",
        action='store_true',
    )
    return parser
