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

# --- 2. 加载元数据 ---
# 【修正】CSV 文件名应为 metadata.csv
data_path = '../data/data.csv' 
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    # 【修正】添加了 f-string 的 'f'
    print(f'no file found in {data_path}.') 
    exit() # 如果文件找不到，直接退出程序

print(f"已加载 {len(df)} 条元数据记录。")

annotation_data = []

# --- 3. 辅助函数 (这个函数是正确的) ---
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

print("开始批量标注关键点...")
# --- 4. 遍历 DataFrame ---
for index, row in tqdm(df.iterrows(),total = df.shape[0]):
    filepath = row['filepath']

    try:
        img = cv.imread(filepath)
        if img is None:
            print(f'\n警告：无法读取图像 {filepath}，已跳过。')
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        rects = detector(gray, 1)

        detection_successful = False
        landmarks_5_points = {} # coordinates of 5 landmarks
        bbox = None

        if len(rects) == 1:
            detection_successful = True
            rect = rects[0]
            
            # --- 【崩溃修正】在这里正确地运行预测器 ---
            shape_68_dlib = predictor(gray, rect)
            landmarks_68_np = shape_to_np(shape_68_dlib)
            landmarks_5_points = get_5_landmarks(landmarks_68_np)
            # --- 修正结束 ---

            # store boundaries
            (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
            bbox = [x, y, x+w, y+h]
        
        elif len(rects) == 0:
            pass # 未检测到，保留为空

        else: # 检测到多张脸
            # 【逻辑修正】这仍然是一次成功的检测
            detection_successful = True 
            biggest_rect = max(rects, key=lambda r:r.width()*r.height())

            shape_68_dlib = predictor(gray, biggest_rect)
            landmarks_68_np = shape_to_np(shape_68_dlib)
            landmarks_5_points = get_5_landmarks(landmarks_68_np)
            (x, y, w, h) = (biggest_rect.left(), biggest_rect.top(), biggest_rect.width(), biggest_rect.height())
            bbox = [x, y, x + w, y + h]
        
        # --- 【缩进修正】这部分必须在 if/elif/else 逻辑块的 *外面* ---
        flat_landmarks = {}
        for name, (point) in landmarks_5_points.items():
            # (确保 point 是 (x, y) 格式)
            if hasattr(point, '__len__') and len(point) == 2:
                flat_landmarks[f'{name}_x'] = point[0]
                flat_landmarks[f'{name}_y'] = point[1]
            else:
                # 处理可能的空值或格式错误
                flat_landmarks[f'{name}_x'] = None
                flat_landmarks[f'{name}_y'] = None
        
        row_data = {
            'filepath': filepath,
            'detection_successful': detection_successful,
            'bbox_x1': bbox[0] if bbox else None,
            'bbox_y1': bbox[1] if bbox else None,
            'bbox_x2': bbox[2] if bbox else None,
            'bbox_y2': bbox[3] if bbox else None,
            **flat_landmarks # 解包展平的 10 个坐标
        }
        annotation_data.append(row_data)
        
    except Exception as e:
        print(f"\n {filepath} unkown error: {e}")

# --- 5. 合并与保存 ---
annotations_df = pd.DataFrame(annotation_data)
final_df = pd.merge(df, annotations_df, on='filepath')

output_path_parquet = "../data/landmarks_dataset.parquet"
# output_path_csv = "..data/landmarks_dataset.csv"
final_df.to_parquet(output_path_parquet, index=False)
# final_df.to_csv(output_path_csv, index=False)

print(f"成功保存标注数据集到: {output_path_parquet}")
print("\n--- 最终数据结构 (列名) ---")
print(final_df.columns.to_list())
print("\n--- 最终数据预览 (前5行) ---")
print(final_df.head())