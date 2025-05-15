import cv2 as cv
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from threading import Thread
from queue import Queue
import time
import os
import sys
from utils import DLT, get_projection_matrix

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_hands   = mp.solutions.hands

# Image and capture params
FRAME_H, FRAME_W = 480, 640
FPS = 30
INTERVAL = 1 / FPS

# Landmark indices
WRIST      = 0
FINGERTIPS = [4, 8, 12, 16, 20]

# Servo mapping params (tweak these to your workspace)
MIN_DIST = 0.05    # meters (or your unit) → fully closed
MAX_DIST = 0.30    # meters → fully open
SERVO_MIN = 0
SERVO_MAX = 180

# Queue for live plotting
data_queue = Queue(maxsize=1)

def simulate_servos(distances):
    """
    Map each fingertip-to-wrist distance to a servo angle.
    Closer → larger angle (closing), farther → smaller angle (opening).
    """
    # invert mapping: dist = MIN_DIST → angle = SERVO_MAX, dist = MAX_DIST → SERVO_MIN
    angles = [
        int(np.clip(
            np.interp(d, [MAX_DIST, MIN_DIST], [SERVO_MIN, SERVO_MAX]),
            SERVO_MIN, SERVO_MAX
        ))
        for d in distances
    ]
    return angles

def live_plot_thread():
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter([], [], [], c='r')
    ax.set_xlim(-200, 200); ax.set_ylim(-200,200); ax.set_zlim(-200,200)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title("Live 3D Hand Keypoints")

    def update(_):
        if not data_queue.empty():
            pts = data_queue.get()
            xs, ys, zs = zip(*pts)
            scatter._offsets3d = (xs, ys, zs)
        return scatter,

    ani = FuncAnimation(fig, update, interval=1000//FPS)
    plt.show()

def run_mp(cam0_idx=1, cam1_idx=2, P0=None, P1=None):
    cap0 = cv.VideoCapture(cam0_idx)
    cap1 = cv.VideoCapture(cam1_idx)
    for c in (cap0, cap1):
        c.set(cv.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        c.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        c.set(cv.CAP_PROP_FPS,         FPS)

    hands0 = mp_hands.Hands(max_num_hands=1, model_complexity=0,
                            min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands1 = mp_hands.Hands(max_num_hands=1, model_complexity=0,
                            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    last_time = time.time()

    while True:
        ret0, f0 = cap0.read()
        ret1, f1 = cap1.read()
        if not ret0 or not ret1:
            break

        # Preprocess
        rgb0 = cv.cvtColor(f0, cv.COLOR_BGR2RGB)
        rgb1 = cv.cvtColor(f1, cv.COLOR_BGR2RGB)
        rgb0.flags.writeable = False
        rgb1.flags.writeable = False

        res0 = hands0.process(rgb0)
        res1 = hands1.process(rgb1)

        # Corrected: Extract 2D landmarks with enumerate()
        lm0 = [[-1, -1]] * 21
        if res0.multi_hand_landmarks:
            hand0 = res0.multi_hand_landmarks[0]
            for idx, lm in enumerate(hand0.landmark):
                lm0[idx] = [int(lm.x * FRAME_W), int(lm.y * FRAME_H)]

        lm1 = [[-1, -1]] * 21
        if res1.multi_hand_landmarks:
            hand1 = res1.multi_hand_landmarks[0]
            for idx, lm in enumerate(hand1.landmark):
                lm1[idx] = [int(lm.x * FRAME_W), int(lm.y * FRAME_H)]

        # Proceed with triangulation, servo simulation, live plotting, etc.
        # ...

        # Triangulate to 3D
        pts3D = []
        for p0, p1 in zip(lm0, lm1):
            if p0[0] < 0 or p1[0] < 0:
                pts3D.append([-1, -1, -1])
            else:
                pts3D.append(DLT(P0, P1, p0, p1))
        pts3D = np.array(pts3D)

        # Compute distances & simulate servos at 30 FPS intervals
        now = time.time()
        if now - last_time >= INTERVAL:
            last_time = now
            if pts3D[WRIST,0] >= 0:
                dists = [np.linalg.norm(pts3D[f] - pts3D[WRIST]) for f in FINGERTIPS]
                angles = simulate_servos(dists)
                print(f"Distances: {[round(d,3) for d in dists]}")
                print(f"Servo angles: {angles}")

                # Send the 6 points (wrist + 5 fingertips) to live plot
                to_plot = [pts3D[WRIST]] + [pts3D[f] for f in FINGERTIPS]
                if data_queue.full():
                    _ = data_queue.get()
                data_queue.put(to_plot)

        # Draw preview
        rgb0.flags.writeable = True
        rgb1.flags.writeable = True
        f0 = cv.cvtColor(rgb0, cv.COLOR_RGB2BGR)
        f1 = cv.cvtColor(rgb1, cv.COLOR_RGB2BGR)
        if res0.multi_hand_landmarks:
            mp_drawing.draw_landmarks(f0, res0.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        if res1.multi_hand_landmarks:
            mp_drawing.draw_landmarks(f1, res1.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        cv.imshow("Cam 0", f0)
        cv.imshow("Cam 1", f1)
        if cv.waitKey(1) & 0xFF == 27:
            break

    cap0.release()
    cap1.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)
    Thread(target=live_plot_thread, daemon=True).start()
    run_mp(1, 2, P0, P1)
