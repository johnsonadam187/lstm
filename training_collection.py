import pandas as pd 
import numpy as np
import os
import time 
from lstm import create_holistic_model, run_process
from io_functions import make_folder


def create_training_folders(num_folders, path, folder_name):
    for num in range(num_folders):
        make_folder(path, folder_name)


def training_data_collection(vid_folder_path, path, action_name, frames=30, training_type="pixel"):
    """data collection process for a single action. requires path of folder containing training vids, path for folder creation for saving data, 
    num_frames for number of frames collected per action, and action name and a training type ("pixel" for pixel value, "angles" for joint anlges) """
    #create new folder for storage
    folder = make_folder(path, action_name)
    #for each video in video folder
    for video in os.listdir(vid_folder_path):
        pass     
#todo finish this by including video functions



if __name__=="__main__":
    pass