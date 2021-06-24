import pandas as pd 
import numpy as np
import os
import time 
from lstm import create_holistic_model, run_process
from io_functions import make_folder


def create_training_folders(num_folders, path, folder_name):
    for num in range(num_folders):
        make_folder(path, folder_name)


def training_data_collection(path, action_name, frames=30):
    """data collection process for a single action. requires path for folder creation, 
    num_frames for number of frames collected per action, and action name """
    folder = make_folder(path, action_name)
    



if __name__=="__main__":
    pass