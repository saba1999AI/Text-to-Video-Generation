import os
import glob
import json
import random
import argparse
from tqdm import tqdm
from loguru import logger
from threading import Lock

import torch
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import snapshot_download
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

file_lock = Lock()

model_path = "../ckpts/data_process/Qwen2-VL-7B-Instruct"

if not os.path.exists(model_path):
    print(f"Model not found, downloading from Hugging Face and Github...")
    snapshot_download(repo_id="Qwen/Qwen2-VL-7B-Instruct", local_dir=model_path)
else:
    print(f"Model already exists in {model_path}, skipping download.")

processor = AutoProcessor.from_pretrained(model_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Process video files in a given directory.")
    parser.add_argument('--input_folder', type=str, default="step1/videos", help="Directory containing video files.")
    parser.add_argument('--output_folder', type=str, default="step1/temp_caption", help="Output directory for processed results.")
    parser.add_argument('--total_gpu', type=int, default=1, help="Total number of gpus.")
    parser.add_argument('--local_gpu_id', type=int, default=0, help="Current gpu id.")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for processing.")
    return parser.parse_args()


class VideoDataset(Dataset):
    def __init__(self, video_files, query, output_folder, processor):
        self.output_folder = output_folder
        self.query = query
        self.video_files = video_files
        self.processor = processor

        fixed_paths = [  
            self.output_folder,  
        ]  

        self.video_files = [  
            video_file for video_file in tqdm(video_files, desc="Filtering video files")  
            if not self._is_processed(video_file, fixed_paths)  
        ]

    def _is_processed(self, video_file, fixed_paths):
        base_name = os.path.splitext(os.path.basename(video_file))[0]
        for path in fixed_paths:
            json_file = os.path.join(path, f"{base_name}.json")
            if os.path.exists(json_file):
                return True
        return False

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        try:
            video_file = self.video_files[idx]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": f"file://{video_file}",
                            "max_pixels": 360 * 420,
                            "fps": 1.0,
                        },
                        {"type": "text", "text": self.query},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_input, video_input = process_vision_info(messages)
            return video_file, text, image_input, video_input
        except Exception as e:
            logger.error(e)
            return self.__getitem__(random.randint(0, self.__len__() - 1))


def collate_fn(batch):
    video_files, texts, image_inputs, video_inputs = zip(*batch)
    video_inputs = [i[0] for i in video_inputs]
    inputs = processor(
        text=list(texts),
        images=None,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return list(video_files), inputs


def process_chunk(total_gpu, local_gpu_id, data):
    chunk_size = len(data) // total_gpu
    start_index = local_gpu_id * chunk_size
    if local_gpu_id == total_gpu - 1:
        end_index = len(data)
    else:
        end_index = start_index + chunk_size

    return data[start_index:end_index]


def load_video_files(input_folder):
    video_files = glob.glob(os.path.join(input_folder, "*.mp4"))
    video_files += glob.glob(os.path.join(input_folder, "*.avi"))
    video_files += glob.glob(os.path.join(input_folder, "*.mkv"))
    
    if not video_files:
        raise ValueError(f"No video files found in {input_folder}")
    
    video_files = [os.path.abspath(file) for file in video_files]
    
    return video_files


def main(args):
    failed_videos_file = 'failed_videos.txt'
    if os.path.exists(failed_videos_file):
        with file_lock:
            with open(failed_videos_file, 'r') as f:
                failed_videos = set(line.strip() for line in f)
    else:
        failed_videos = set()

    os.makedirs(args.output_folder, exist_ok=True)

    video_files = load_video_files(args.input_folder)

    query = "Please describe the video in detail"
    random.shuffle(video_files)
    video_files = process_chunk(args.total_gpu, args.local_gpu_id, video_files)
    dataset = VideoDataset(video_files, query, args.output_folder, processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, collate_fn=collate_fn)

    if len(dataset) == 0:
        print(f"len dataset: {len(dataset)}, skip")
        return
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        model = model.eval()

    for video_files, inputs in tqdm(dataloader):
        filtered_video_files = [vf for vf in video_files if vf not in failed_videos]
        if not filtered_video_files:
            continue

        video_files = filtered_video_files

        try:
            inputs = inputs.to("cuda")

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=512)
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            for video_file, output_text in zip(video_files, output_texts):
                output_file = os.path.join(args.output_folder, f"{os.path.splitext(os.path.basename(video_file))[0]}.json")
                with open(output_file, "w") as f:
                    json.dump({"video": os.path.basename(video_file).replace('.mp4', ''), "description": output_text}, f, indent=4)
                logger.info(f"Processed {video_file}, saved to {output_file}")
        except Exception as e:
            logger.error(f"Error processing {video_files}: {e}")
            with file_lock:
                with open(failed_videos_file, 'a') as f:
                    for video in video_files:
                        f.write(video + '\n')
            torch.cuda.empty_cache()

    logger.warning(f"All videos processed in {args.local_gpu_id}.")


if __name__ == '__main__':
    args = parse_args()
    main(args)