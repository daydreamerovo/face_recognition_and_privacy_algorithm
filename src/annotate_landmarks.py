import dlib
import cv2 as cv
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from imutils.face_utils import shape_to_np


model_path = '../models/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat' 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)


data_path = '../data/data.csv' 
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print(f'no file found in {data_path}.') 
    exit() 

print(f"loaded {len(df)} records.")

annotation_data = []

def get_5_landmarks(landmarks_68):
    """
    extract left/right eye, left/right mouth corner, and nose landmarks.
    """
    right_eye_pts = landmarks_68[36:42]
    right_eye_center = np.mean(right_eye_pts, axis=0).astype(int)

    left_eye_pts = landmarks_68[42:48]
    left_eye_center = np.mean(left_eye_pts, axis=0).astype(int)

    nose_tip = landmarks_68[30]
    right_mouth_corner = landmarks_68[48]
    left_mouth_corner = landmarks_68[54]

    return {
        'right_eye':right_eye_center,
        'left_eye':left_eye_center,
        'nose_tip':nose_tip,
        'right_mouth':right_mouth_corner,
        'left_mouth':left_mouth_corner
    }


for index, row in tqdm(df.iterrows(),total = df.shape[0]):
    filepath = row['filepath']

    try:
        img = cv.imread(filepath)
        if img is None:
            print(f'\ncannot read{filepath}, skipped.')
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        rects = detector(gray, 1)

        detection_successful = False
        landmarks_5_points = {} # coordinates of 5 landmarks
        bbox = None

        if len(rects) == 1:
            detection_successful = True
            rect = rects[0]
            
            shape_68_dlib = predictor(gray, rect)
            landmarks_68_np = shape_to_np(shape_68_dlib)
            landmarks_5_points = get_5_landmarks(landmarks_68_np)

            # store boundaries
            (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
            bbox = [x, y, x+w, y+h]
        
        elif len(rects) == 0:
            pass 

        else: # multiple faces
            detection_successful = True 
            biggest_rect = max(rects, key=lambda r:r.width()*r.height())

            shape_68_dlib = predictor(gray, biggest_rect)
            landmarks_68_np = shape_to_np(shape_68_dlib)
            landmarks_5_points = get_5_landmarks(landmarks_68_np)
            (x, y, w, h) = (biggest_rect.left(), biggest_rect.top(), biggest_rect.width(), biggest_rect.height())
            bbox = [x, y, x + w, y + h]
        
        flat_landmarks = {}
        for name, (point) in landmarks_5_points.items():
            if hasattr(point, '__len__') and len(point) == 2:
                flat_landmarks[f'{name}_x'] = point[0]
                flat_landmarks[f'{name}_y'] = point[1]
            else:
                flat_landmarks[f'{name}_x'] = None
                flat_landmarks[f'{name}_y'] = None
        
        row_data = {
            'filepath': filepath,
            'detection_successful': detection_successful,
            'bbox_x1': bbox[0] if bbox else None,
            'bbox_y1': bbox[1] if bbox else None,
            'bbox_x2': bbox[2] if bbox else None,
            'bbox_y2': bbox[3] if bbox else None,
            **flat_landmarks
        }
        annotation_data.append(row_data)
        
    except Exception as e:
        print(f"\n {filepath} unkown error: {e}")

annotations_df = pd.DataFrame(annotation_data)
final_df = pd.merge(df, annotations_df, on='filepath')

output_path_parquet = "../data/landmarks_dataset.parquet"
# output_path_csv = "..data/landmarks_dataset.csv"
final_df.to_parquet(output_path_parquet, index=False)
# final_df.to_csv(output_path_csv, index=False)

print(f"file saved at:{output_path_parquet}")
print("\n column")
print(final_df.columns.to_list())
print("\n row")
print(final_df.head())