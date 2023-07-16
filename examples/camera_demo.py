import time
import cv2
import numpy as np
from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints


def detect_raise_hand(keypoints):
    right_shoulder, right_elbow, right_wrist = keypoints[0, 5].cpu().numpy(), keypoints[0, 6].cpu().numpy(), keypoints[0, 7].cpu().numpy()
    left_shoulder, left_elbow, left_wrist = keypoints[0, 2].cpu().numpy(), keypoints[0, 3].cpu().numpy(), keypoints[0, 4].cpu().numpy()

    # Checking if clapping
    if np.linalg.norm(right_wrist - left_wrist) < min(np.linalg.norm(right_wrist - right_elbow), np.linalg.norm(left_wrist - left_elbow)):
        return "Clapping"

    # Checking if the right hand is raised
    if np.linalg.norm(right_shoulder - right_wrist) < np.linalg.norm(right_shoulder - right_elbow):
        return "Right hand is raised"

    # Checking if the left hand is raised
    if np.linalg.norm(left_shoulder - left_wrist) < np.linalg.norm(left_shoulder - left_elbow):
        return "Left hand is raised"

    return None


estimator = BodyPoseEstimator(pretrained=True)
camera = cv2.VideoCapture(0)

frame_count = 0
start_time = time.time()

while camera.isOpened():
    flag, frame = camera.read()
    if not flag:
        break

    frame_count += 1

    # Resize the frame to speed up processing
    frame = cv2.resize(frame, (640, 480))
    keypoints = estimator(frame)
    frame = draw_body_connections(frame, keypoints, thickness=2, alpha=0.7)
    frame = draw_keypoints(frame, keypoints, radius=4, alpha=0.8)

    # Detect gestures
    gesture = detect_raise_hand(keypoints)
    if gesture:
        cv2.putText(frame, 'Gesture : ' + str(gesture), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw index of each keypoint next to it
    keypoints_np = keypoints[0].cpu().numpy()
    for i, keypoint in enumerate(keypoints_np):
        # Draw the index of the keypoint
        cv2.putText(frame, str(i), (int(keypoint[0]), int(keypoint[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


    # Calculate and display FPS
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(frame, 'FPS : ' + str(fps), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video Demo', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):  # exit if pressed `q`
        break

camera.release()
cv2.destroyAllWindows()
