import cv2
import os
from matplotlib import pyplot as plt
import time 
import timeit
import mediapipe as mp
from lstm import pose_detection_results, flatten_pose_results, create_holistic_model


def train_from_video(path, model, frames, training_length=1):
    """allows training for single item from video file, requires video path, 
    mediapipe training model, frames per second for sampling, and the training time length for sampling"""
    data_collection = []
    cap = cv2.VideoCapture(path)
    cap.set(5, frames)
    start = time.time()
    finish = start + training_length
    while time.time() < finish:
        ret, frame = cap.read()
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results =  pose_detection_results(rgb_image, model)
        flat = flatten_pose_results(results)
        data_collection.append(flat)
    cap.release()
    cv2.destroyAllWindows()
    return data_collection
    

def train_from_camera(webcam_number, model, frames, training_length=1):
    """allows training for single item from video cam, requires webcam number, 
    mediapipe training model, frames per second for sampling, and the training time length for sampling"""
    data_collection = []
    cap = cv2.VideoCapture(webcam_number)
    cap.set(5, frames)
    start = time.time()
    finish = start + training_length
    while time.time() < finish:
        ret, frame = cap.read()
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("Training", frame)
        results =  pose_detection_results(rgb_image, model)
        flat = flatten_pose_results(results)
        data_collection.append(flat)
    cap.release()
    cv2.destroyAllWindows()
    return data_collection


def time_tester():
    t1 = time.time()
    t2 = t1 + 1
    print(t1)
    while time.time() < t2:
        print(t2)



if __name__=="__main__":
    model = create_holistic_model(min_detection_confidence=0.5, min_tracking_confidence=0.5)    
    results = train_from_camera(0, model, 1, training_length=2)
    print(len(results), len(results[0]))  


# TODO finish video testers by sorting uniformity in framerate