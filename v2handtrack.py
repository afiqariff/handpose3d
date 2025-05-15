import cv2 as cv
import mediapipe as mp
import numpy as np
import csv
import time
import os
import sys
from utils import DLT, get_projection_matrix, write_keypoints_to_disk

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Fingertip landmarks and wrist
FINGERTIP_IDS = [4, 8, 12, 16, 20]  # Thumb_tip, Index_tip, etc.
WRIST_ID = 0

# Target resolution
frame_shape = [480, 640]  # Height x Width

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def run_mp(input_stream1, input_stream2, P0, P1):
    cap0 = cv.VideoCapture(input_stream1)
    cap1 = cv.VideoCapture(input_stream2)
    caps = [cap0, cap1]

    for cap in caps:
        cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_shape[1])
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_shape[0])
        cap.set(cv.CAP_PROP_FPS, 30)

    hands0 = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=1, min_tracking_confidence=0.5)
    hands1 = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=1, min_tracking_confidence=0.5)

    kpts_cam0, kpts_cam1, kpts_3d = [], [], []
    csv_data = []
    frame_count = 0
    start_time = time.time()

    ensure_dir("images/cam0")
    ensure_dir("images/cam1")

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            break

        frame0_rgb = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
        frame1_rgb = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)

        frame0_rgb.flags.writeable = False
        frame1_rgb.flags.writeable = False
        results0 = hands0.process(frame0_rgb)
        results1 = hands1.process(frame1_rgb)

        frame0_keypoints, frame1_keypoints = [], []

        # Frame 0 keypoints
        if results0.multi_hand_landmarks:
            for hand_landmarks in results0.multi_hand_landmarks:
                for p in range(21):
                    x = int(round(frame0.shape[1]*hand_landmarks.landmark[p].x))
                    y = int(round(frame0.shape[0]*hand_landmarks.landmark[p].y))
                    frame0_keypoints.append([x, y])
        else:
            frame0_keypoints = [[-1, -1]] * 21

        # Frame 1 keypoints
        if results1.multi_hand_landmarks:
            for hand_landmarks in results1.multi_hand_landmarks:
                for p in range(21):
                    x = int(round(frame1.shape[1]*hand_landmarks.landmark[p].x))
                    y = int(round(frame1.shape[0]*hand_landmarks.landmark[p].y))
                    frame1_keypoints.append([x, y])
        else:
            frame1_keypoints = [[-1, -1]] * 21

        kpts_cam0.append(frame0_keypoints)
        kpts_cam1.append(frame1_keypoints)

        # Triangulate 3D keypoints
        frame_p3ds = []
        for uv1, uv2 in zip(frame0_keypoints, frame1_keypoints):
            if uv1[0] == -1 or uv2[0] == -1:
                p3d = [-1, -1, -1]
            else:
                p3d = DLT(P0, P1, uv1, uv2)
            frame_p3ds.append(p3d)

        frame_p3ds = np.array(frame_p3ds).reshape((21, 3))
        kpts_3d.append(frame_p3ds)

        # Extract wrist and fingertips
        if all(frame_p3ds[idx][0] != -1 for idx in [WRIST_ID] + FINGERTIP_IDS):
            wrist = frame_p3ds[WRIST_ID]
            distances = []
            for fid in FINGERTIP_IDS:
                finger = frame_p3ds[fid]
                dist = np.linalg.norm(np.array(finger) - np.array(wrist))
                distances.append(dist)

            row = [frame_count] + wrist.tolist()
            for fid in FINGERTIP_IDS:
                row += frame_p3ds[fid].tolist()
            row += distances
            csv_data.append(row)

        # Draw landmarks
        frame0_rgb.flags.writeable = True
        frame1_rgb.flags.writeable = True
        frame0 = cv.cvtColor(frame0_rgb, cv.COLOR_RGB2BGR)
        frame1 = cv.cvtColor(frame1_rgb, cv.COLOR_RGB2BGR)

        if results0.multi_hand_landmarks:
            for hl in results0.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame0, hl, mp_hands.HAND_CONNECTIONS)
        if results1.multi_hand_landmarks:
            for hl in results1.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame1, hl, mp_hands.HAND_CONNECTIONS)

        # Save images at ~30 FPS
        if time.time() - start_time >= 1 / 30:
            cv.imwrite(f'images/cam0/frame_{frame_count:04d}.jpg', frame0)
            cv.imwrite(f'images/cam1/frame_{frame_count:04d}.jpg', frame1)
            frame_count += 1
            start_time = time.time()

        cv.imshow('cam0', frame0)
        cv.imshow('cam1', frame1)

        if cv.waitKey(1) & 0xFF == 27:
            break

    for cap in caps:
        cap.release()
    cv.destroyAllWindows()

    # Save keypoints for further use
    write_keypoints_to_disk('kpts_cam0.dat', kpts_cam0)
    write_keypoints_to_disk('kpts_cam1.dat', kpts_cam1)
    write_keypoints_to_disk('kpts_3d.dat', kpts_3d)

    # Write CSV for matplotlib
    headers = ['Frame', 'Wrist_X', 'Wrist_Y', 'Wrist_Z']
    for fid in FINGERTIP_IDS:
        headers += [f'F{fid}_X', f'F{fid}_Y', f'F{fid}_Z']
    headers += [f'Dist_F{fid}_to_Wrist' for fid in FINGERTIP_IDS]

    with open("hand_3d_data.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(csv_data)

    return np.array(kpts_cam0), np.array(kpts_cam1), np.array(kpts_3d)

if __name__ == '__main__':
    input_stream1 = 1
    input_stream2 = 2

    if len(sys.argv) == 3:
        input_stream1 = int(sys.argv[1])
        input_stream2 = int(sys.argv[2])

    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)

    run_mp(input_stream1, input_stream2, P0, P1)
