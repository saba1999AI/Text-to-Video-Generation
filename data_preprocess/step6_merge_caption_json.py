import os
import cv2
import json
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Process video files in a given directory.")
    parser.add_argument('--input_video_folder', type=str, default="step1/videos", help="Path to the input video folder")
    parser.add_argument('--temp_caption_folder', type=str, default='step1/temp_caption', help="Path to the temporary caption folder")
    parser.add_argument('--valid_json_folder', type=str, default='step1/refine_bbox_jsons', help="Path to the folder containing valid JSON files")
    parser.add_argument('--output_caption_path', type=str, default='step1/captions/merge_caption.json', help="Path to save the merged caption file")
    return parser.parse_args()


def merge_valid_caption(valid_json_dir, temp_caption_folder, output_caption_path):
    valid_files = [f for f in os.listdir(valid_json_dir) if f.endswith('.json')]

    merged_data = []

    for filename in tqdm(valid_files):
        all_json_path = os.path.join(temp_caption_folder, filename)
        if os.path.exists(all_json_path):
            try:
                with open(all_json_path, 'r') as f:
                    data = json.load(f)
                    merged_data.append(data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from file {all_json_path}: {e}")
            except Exception as e:
                print(f"Unexpected error reading file {all_json_path}: {e}")
        else:
            print(f"File {filename} not found in {temp_caption_folder}.")

    try:
        with open(output_caption_path, 'w') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error writing to output file {output_caption_path}: {e}")


def update_video_metadata(input_caption_path, input_video_folder):
    with open(input_caption_path, 'r') as file:
        videos = json.load(file)
    
    for video in tqdm(videos):
        if 'video' in video:
            video['path'] = video.pop('video')

        if 'description' in video:
            video['cap'] = video.pop('description')

        video_path = os.path.join(input_video_folder, f"{video['path']}.mp4")
        
        if os.path.exists(video_path):
            video['size'] = os.path.getsize(video_path)
        else:
            print(f"Warning: The file {video_path} does not exist.")
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open the video file {video_path}")
            continue
        
        video['resolution'] = {
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))            
        }
        
        video['fps'] = cap.get(cv2.CAP_PROP_FPS)
        video['duration'] = cap.get(cv2.CAP_PROP_FRAME_COUNT) / video['fps']
        cap.release()

    with open(input_caption_path, 'w') as file:
        json.dump(videos, file, indent=4)
        print(f"Merged JSON has been saved to {output_caption_path}.")


if __name__ == "__main__":
    args = parse_args()

    input_video_folder = args.input_video_folder
    temp_caption_folder = args.temp_caption_folder
    valid_json_folder   = args.valid_json_folder
    output_caption_path = args.output_caption_path
    output_folder = os.path.dirname(output_caption_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    merge_valid_caption(valid_json_folder, temp_caption_folder, output_caption_path)
    update_video_metadata(output_caption_path, input_video_folder)