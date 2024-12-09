import cv2
import json

annotations_path = 'your.json'
input_video_path = 'your.mp4'
output_video_path = 'output_bbox.mp4'


def draw_boxes(frame, annotations):
    # Draw bounding boxes for faces
    for face in annotations.get("face", []):
        box = face["box"]
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        track_id = face.get("new_track_id", None)
        label = f'{face["name"]}: {face["confidence"]:.2f}'
        if track_id is not None:
            label += f' (ID: {track_id})'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Draw bounding boxes for heads
    for head in annotations.get("head", []):
        box = head["box"]
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        track_id = head.get("new_track_id", None)
        label = f'{head["name"]}: {head["confidence"]:.2f}'
        if track_id is not None:
            label += f' (ID: {track_id})'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Draw bounding boxes for persons
    for person in annotations.get("person", []):
        box = person["box"]
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        track_id = person.get("new_track_id", None)
        label = f'{person["name"]}: {person["confidence"]:.2f}'
        if track_id is not None:
            label += f' (ID: {track_id})'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return frame


def process_video(input_video_path, output_video_path, annotations_path):
    # Load the annotations from the JSON file
    with open(annotations_path, 'r') as file:
        annotations = json.load(file)

    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get the annotations for the current frame
        frame_annotations = annotations.get(str(frame_count), {})

        # Draw the boxes on the frame
        frame = draw_boxes(frame, frame_annotations)

        # Write the frame to the output video
        out.write(frame)
        frame_count += 1

        if frame_count % 100 == 0:
            print(f'Processed {frame_count}/{total_frames} frames')

    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f'Processing complete. Output saved to {output_video_path}')


process_video(input_video_path, output_video_path, annotations_path)