import pandas as pd 
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import time 
import mediapipe as mp

def convert_image(image, conversion):
    img2 = cv2.cvtColor(image, conversion)
    return img2


def pose_detection_results(image, mp_model):
    image.flags.writeable = False
    pose_results = mp_model.process(image)
    image.flags.writeable = True
    return pose_results


def create_holistic_model(mp_class=mp.solutions.holistic.Holistic,  **kwargs):
    mp_model = mp_class(kwargs)
    return mp_model


def draw_holistic_pose(image, results):
    mp_draw = mp.solutions.drawing_utils
    mp_draw.draw_landmarks(image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)      


def flatten_pose_results(results):
    flattened = np.array([[res.x, res.y, res.z, res.visibility]for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    return flattened



def run_process(mp_model):
    with mp_model as pose_detection_model:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            black_and_white = convert_image(frame, cv2.COLOR_BGR2GRAY)
            rgb_image = convert_image(frame, cv2.COLOR_BGR2RGB)
            pose_points = pose_detection_results(rgb_image, pose_detection_model)
            draw_holistic_pose(frame, pose_points)
            # print(pose_points)
            cv2.imshow("Camera Feed", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()



if __name__=="__main__":
    pose_model = create_holistic_model(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    run_process(pose_model)