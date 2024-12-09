import os
import json
import argparse
import itertools
from tqdm import tqdm
import multiprocessing
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from util.download_weights import download_file


def parse_args():
    parser = argparse.ArgumentParser(description="Process MP4 files with YOLO models.")
    parser.add_argument('--input_video_folder', type=str, default='step0/videos', help='Path to the folder containing MP4 files.')
    parser.add_argument('--output_json_folder', type=str, default='step0/json', help='Directory for output files.')
    parser.add_argument("--model_path", type=str, default="../ckpts", help="The path of the pre-trained model to be used")
    parser.add_argument('--num_processes', type=int, default=1, help='Number of processes to use.')
    return parser.parse_args()


def process_mp4_files(mp4_files_chunk, args):
    print(f"Process {os.getpid()} is handling {len(mp4_files_chunk)} files.")
    
    output_json_folder = args.output_json_folder
    face_model = YOLO(os.path.join(args.model_path, "data_process", "step1_yolov8_face.pt")).cuda()
    head_model = YOLO(os.path.join(args.model_path, "data_process", "step1_yolov8_head.pt")).cuda()
    person_model = YOLO(os.path.join(args.model_path, "data_process", "yolov8l-worldv2.pt")).cuda()
    pose_model = YOLO(os.path.join(args.model_path, "data_process", "yolov8l-pose.pt")).cuda()
    person_model.set_classes(["person"])

    for source in tqdm(mp4_files_chunk, desc="Processing files"):
        save_name = os.path.basename(source).replace('.mp4', '.json')
        save_path = os.path.join(output_json_folder, save_name)

        if os.path.exists(save_path):
            print(f"Skipping {save_name}")
            continue
        
        video_detect_results = {}
        results_face = face_model.track(source, stream=True, conf=0.5)
        results_head = head_model.track(source, stream=True, conf=0.6)
        results_person = person_model.track(source, stream=True, conf=0.6)
        results_pose = pose_model.track(source, stream=True)

        try:
            for frame_idx, (result_face, result_head, result_person, result_pose) in enumerate(itertools.zip_longest(results_face, results_head, results_person, results_pose, fillvalue=None)):
                video_detect_results[frame_idx] = {
                    'face': json.loads(result_face.to_json()) if result_face else [],
                    'head': json.loads(result_head.to_json()) if result_head else [],
                    'person': json.loads(result_person.to_json()) if result_person else [],
                    'pose': json.loads(result_pose.to_json()) if result_pose else []
                }
        except Exception as e:
            print(f"Error processing {source}: {e}")
        
        final_json_str = json.dumps(video_detect_results, indent=4)
        with open(save_path, 'w') as json_file:
            json_file.write(final_json_str)


def main():
    args = parse_args()
    
    model_files = [
        "step1_yolov8_face.pt",
        "step1_yolov8_head.pt",
        "yolov8l-worldv2.pt",
        "yolov8l-pose.pt",
    ]
    
    if not any(os.path.exists(os.path.join(args.model_path, "data_process", file)) for file in model_files):
        print(f"Model not found, downloading from Hugging Face and Github...")
        hf_hub_download(repo_id="BestWishYsh/ConsisID-preview", filename="data_process/step1_yolov8_face.pt", local_dir=args.model_path)
        hf_hub_download(repo_id="BestWishYsh/ConsisID-preview", filename="data_process/step1_yolov8_head.pt", local_dir=args.model_path)
        download_file("https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-worldv2.pt", os.path.join(args.model_path, "data_process", "yolov8l-worldv2.pt"))
        download_file("https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-pose.pt", os.path.join(args.model_path, "data_process", "yolov8l-pose.pt"))
    else:
        print(f"Model already exists in {args.model_path}, skipping download.")

    os.makedirs(args.output_json_folder, exist_ok=True)
    
    mp4_files = [os.path.join(args.input_video_folder, f) for f in os.listdir(args.input_video_folder) if f.endswith('.mp4')]
    mp4_files_chunks = [mp4_files[i::args.num_processes] for i in range(args.num_processes)]

    processes = []
    for chunk in mp4_files_chunks:
        p = multiprocessing.Process(target=process_mp4_files, args=(chunk, args))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()