import time
import cv2
from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints

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

    # Calculate and display FPS
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(frame, 'FPS : ' + str(fps), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video Demo', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):  # exit if pressed `q`
        break

camera.release()
cv2.destroyAllWindows()
