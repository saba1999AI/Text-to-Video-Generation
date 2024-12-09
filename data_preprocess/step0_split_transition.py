
import os
import shutil
import argparse
import threading
from tqdm import tqdm
from scenedetect.detectors import AdaptiveDetector
from scenedetect import open_video, SceneManager, split_video_ffmpeg
from concurrent.futures import ThreadPoolExecutor, as_completed

file_lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser(description="Process video files in a given directory.")
    parser.add_argument('--input_video_folder', type=str, default="../asserts/demo_train_data/dataname/videos", help="Directory containing video files (e.g., .mp4 files)")
    parser.add_argument('--output_video_folder', type=str, default="step0/videos", help="Directory for output videos")
    parser.add_argument('--processed_videos_file', type=str, default="processed_videos.txt", help="File to keep track of processed videos")
    parser.add_argument('--num_processes', type=int, default=1, help="Max number of parallel workers")
    return parser.parse_args()


def extract_start_end_frames(scene_list):
    start_end_frames = []
    for segment in scene_list:
        start_info, end_info = segment
        start_frame = start_info.frame_num
        end_frame = end_info.frame_num - 1
        start_end_frames.append((start_frame, end_frame))
    return start_end_frames


def split_video_into_scenes(video_path, output_video_folder, processed_videos, processed_videos_file):
    video_name = os.path.basename(video_path).replace('.mp4', '')

    if video_name in processed_videos:
        print(f"Video {video_name} has already been processed. Skipping...")
        return

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(AdaptiveDetector())
    scene_manager.detect_scenes(video, show_progress=False)
    scene_list = scene_manager.get_scene_list()
    scene_frame_list = extract_start_end_frames(scene_list)

    output_files = []

    if not scene_list:
        output_video_path = os.path.join(output_video_folder, f"{video_name}.mp4")
        shutil.copy(video_path, output_video_path)
        output_files.append(output_video_path)
    else:
        for i, (start_frame, end_frame) in enumerate(scene_frame_list):
            output_video_path = os.path.join(output_video_folder, f"{video_name}-Scene-{i+1:03d}.mp4")
            split_video_ffmpeg(video_path, scene_list, output_video_folder, output_file_template='$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4', show_progress=False)
            output_files.append(output_video_path)
        
    all_files_exist = all([os.path.exists(file) for file in output_files])

    if all_files_exist:
        with file_lock:
            processed_videos.add(video_name)
            with open(processed_videos_file, 'a') as f:
                f.write(f"{video_name}\n")
    else:
        print(f"Some files for video {video_name} were not created successfully. Skipping update to processed_videos.")


def process_videos_in_parallel(video_paths, processed_videos, processed_videos_file, output_video_folder, num_processes=32):
    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        future_to_video = {executor.submit(split_video_into_scenes, video_path, output_video_folder, processed_videos, processed_videos_file): video_path for video_path in video_paths}

        for future in tqdm(as_completed(future_to_video), total=len(video_paths), desc="Processing videos"):
            video_path = future_to_video[future]
            try:
                future.result()
            except Exception as exc:
                print(f"{video_path} generated an exception: {exc}")


def main():
    args = parse_args()

    input_video_folder = args.input_video_folder
    output_video_folder = args.output_video_folder
    processed_videos_file = args.processed_videos_file
    num_processes = args.num_processes

    video_paths = [os.path.join(input_video_folder, f) for f in os.listdir(input_video_folder) if f.endswith('.mp4')]
    os.makedirs(output_video_folder, exist_ok=True)

    if os.path.exists(processed_videos_file):
        with open(processed_videos_file, 'r') as f:
            processed_videos = set(f.read().splitlines())
    else:
        processed_videos = set()

    process_videos_in_parallel(video_paths, processed_videos, processed_videos_file, output_video_folder, num_processes)


if __name__ == "__main__":
    main()