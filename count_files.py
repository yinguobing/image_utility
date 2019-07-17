"""
This script shows how to count all files in a specific directory.
"""

from argparse import ArgumentParser
import os
from collections import Counter

parser = ArgumentParser()
parser.add_argument('dir', type=str, help='target path')
args = parser.parse_args()


def get_extention(file_name=None):
    """
    Return the file name extention, or None if the file doesn't have one.
    """
    crumbs = file_name.split(".")
    crumbs_num = len(crumbs)
    if crumbs_num == 1:
        return None
    else:
        return crumbs[-1]


def count_files(directory=None):
    """
    Count all files in directory, and return the dict contains the result.
    """
    file_extentions = []
    none_extentions_num = 0
    for _, _, files in os.walk(directory):
        for file in files:
            extention = get_extention(file)
            if extention is None:
                none_extentions_num += 1
            else:
                file_extentions.append(extention)
    ext_counter = Counter(file_extentions)
    if none_extentions_num != 0:
        ext_counter.update({"None": none_extentions_num})
    return ext_counter


def main():
    """
    The main entrance.
    """
    extention_dict = dict(count_files(args.dir))
    total_count = sum(extention_dict.values())

    print("Total files:", total_count)
    print(extention_dict)
    print("Done!")


if __name__ == '__main__':
    main()
