import os
import cv2
import pickle
from tqdm import tqdm
from argparse import ArgumentParser
from head_pose_estimation.mark_detector import MarkDetector
from head_pose_estimation.pose_estimator import PoseEstimator


# Parse arguments from user input.
parser = ArgumentParser()
parser.add_argument("--data_root", type=str, required=True,
                    help="Casual dataset root to be processed.")
parser.add_argument("--save_root", type=int, required=True,
                    help="Root to save processed dataset")
args = parser.parse_args()


for file_name in sorted(os.listdir(root)):
    if not file_name.endswith('zip'):
        continue
    if not os.path.exists(os.path.join(save_root, file_name)):
        command = f'unzip {os.path.join(root, file_name)} -d {save_root}'
        os.system(command)

    for dir_name in os.listdir(save_root):
        if not dir_name.startswith('CasualConversations'):
            continue
        print(f'process on {file_name} {dir_name}')
        os.makedirs(os.path.join(save_root, 'metadata', dir_name), exist_ok=True)

        for video_dir in tqdm(os.listdir(os.path.join(save_root, dir_name))):
            os.makedirs(os.path.join(save_root, 'metadata', dir_name, video_dir), exist_ok=True)

            for video_name in os.listdir(os.path.join(save_root, dir_name, video_dir)):
                video_path = os.path.join(save_root, dir_name, video_dir, video_name)
                cap = cv2.VideoCapture(video_path)

                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

                pose_estimator = PoseEstimator(img_size=(height, width))
                mark_detector = MarkDetector()

                tm = cv2.TickMeter()

                video_infos = []
                idx = 0
                while True:
                    # Read a frame.
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Step 1: Get a face from current frame.
                    facebox = mark_detector.extract_cnn_facebox(frame)

                    # Any face found?
                    if facebox is not None:
                        # print(f'frame {idx}')
                        # print(frame.shape)
                        # print(f'facebox: {facebox}')

                        # Step 2: Detect landmarks. Crop and feed the face area into the
                        # mark detector.
                        x1, y1, x2, y2 = facebox
                        face_img = frame[y1: y2, x1: x2]

                        # Run the detection.
                        tm.start()
                        marks = mark_detector.detect_marks(face_img)
                        tm.stop()

                        # Convert the locations from local face area to the global image.
                        marks *= (x2 - x1)
                        marks[:, 0] += x1
                        marks[:, 1] += y1

                        # print(f'facial landmarks:\n{marks}')

                        # Try pose estimation with 68 points.
                        pose = pose_estimator.solve_pose_by_68_points(marks)

                        # print(f'pose:\n{pose}')

                        video_infos.append({
                            'index': idx,
                            'facebox': facebox,
                            'landmarks': marks,
                            'pose': pose
                        })

                        # All done. The best way to show the result would be drawing the
                        # pose on the frame in realtime.

                        # Do you want to see the pose annotation?
                        # pose_estimator.draw_annotation_box(
                        #     frame, pose[0], pose[1], color=(0, 255, 0))

                        # Do you want to see the head axes?
                        # pose_estimator.draw_axes(frame, pose[0], pose[1])

                        # Do you want to see the marks?
                        # mark_detector.draw_marks(frame, marks, color=(0, 255, 0))

                        # Do you want to see the facebox?
                        # mark_detector.draw_box(frame, [facebox])

                        # cv2.imwrite('temp.jpg', frame)

                    idx += 1
                with open(os.path.join(save_root, 'metadata', dir_name, video_dir, video_name[:-4]+'.pickle'), "wb") as f:
                    pickle.dump(video_infos, f)

        # Remove files
        command = f'rm -r {os.path.join(save_root, dir_name)}'
        os.system(command)
