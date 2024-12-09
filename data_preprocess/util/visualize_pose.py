import cv2
import json

# Constants
annotations_path = 'your.json'
input_video_path = 'your.mp4'
output_video_path = 'output_pose.mp4'


def draw_pose(frame, annotations):
    for annotation in annotations:
        # Draw bounding box
        box = annotation['box']
        cv2.rectangle(frame, 
                      (int(box['x1']), int(box['y1'])), 
                      (int(box['x2']), int(box['y2'])), 
                      (0, 255, 0), 2)  # Green for bbox
        
        # Draw keypoints
        keypoints = annotation['keypoints']
        for x, y, v in zip(keypoints['x'], keypoints['y'], keypoints['visible']):
            if v > 0.5:  # Only draw visible keypoints
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)  # Red for keypoints

    return frame


def process_video(input_video_path, annotations_path, output_video_path):
    # Open input video
    cap = cv2.VideoCapture(input_video_path)

    with open(annotations_path, 'r') as f:
        data = json.load(f)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if str(frame_idx) in data:
            annotations = data[str(frame_idx)]['pose']
            frame = draw_pose(frame, annotations)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f'Processing complete. Output saved to {output_video_path}')


process_video(input_video_path, annotations_path, output_video_path)