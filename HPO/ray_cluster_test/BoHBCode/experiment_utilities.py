# Copyright (c) [2024] [Dipti Sengupta]
# Licensed under the CC0 1.0 Universal See LICENSE file in the project root for full license information.

"""
Functions that are used to run experiments in the BoHB framework.
"""
import os
import glob
import fnmatch

def remove_checkpoint_files(working_dir):
    """
    Remove all files in a given directory that have the extension '.ckpt'.
    :param checkpoint_dir: The directory with the runs.
    """
    for dirpath, dirs, files in os.walk(working_dir):  
        for filename in fnmatch.filter(files, '*.ckpt'): 
            os.remove(os.path.join(dirpath, filename))
            print(f"Removed file: {os.path.join(dirpath, filename)}")

