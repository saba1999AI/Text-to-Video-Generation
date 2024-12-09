import os
import gc
import cv2
import json
import torch
import shutil
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import supervision as sv
from threading import Lock
from huggingface_hub import hf_hub_download

from sam2.build_sam import build_sam2_video_predictor

file_lock = Lock()

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def parse_args():
    parser = argparse.ArgumentParser(description="Process video files in a given directory.")
    parser.add_argument("--json_folder", type=str, default="step1/refine_bbox_jsons", help="Folder containing JSON files.")
    parser.add_argument("--video_folder", type=str, default="step1/videos", help="Folder containing video files.")
    parser.add_argument("--output_path", type=str, default="step1/track_masks_data", help="Path to store the output files.")
    parser.add_argument("--sam2_checkpoint_path", type=str, default="../ckpts/data_process/", help="Path to the SAM2 model checkpoint.")
    parser.add_argument("--model_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml", help="Path to the SAM2 model configuration.")
    return parser.parse_args()


def estimate_num_people(data):
    max_people_PlanA = 0
    max_people_PlanB = 0
    max_people_PlanC = 0
    for frame_id, objects in data.items():
        num_persons = len(objects.get('person', []))
        num_heads = len(objects.get('head', []))
        num_faces = len(objects.get('face', []))
        if num_persons == num_heads == num_faces:
            max_people_PlanA = max(max_people_PlanA, num_persons)
        if num_persons == num_heads:
            max_people_PlanB = max(max_people_PlanB, num_persons)
        if num_persons == num_faces:
            max_people_PlanC = max(max_people_PlanC, num_persons)
    if max_people_PlanA != 0:
        return max_people_PlanA
    elif max_people_PlanB != 0:
        return max_people_PlanB
    else:
        return max_people_PlanC


def find_max_confidence_bbox(data, max_people, confidence_threshold=0.85):
    best_appearance = {}
    head_to_face_mapping = {}

    for frame_id, objects in data.items():
        for object_type in ['face', 'head', 'person']:
            for item in objects.get(object_type, []):
                try:
                    new_track_id = item['new_track_id']
                except:
                    continue
                confidence = item['confidence']
                if new_track_id <= max_people and confidence > confidence_threshold:
                    if new_track_id not in best_appearance:
                        best_appearance[new_track_id] = {}
                    
                    if (object_type not in best_appearance[new_track_id] or
                        best_appearance[new_track_id][object_type]['confidence'] < confidence):
                        best_appearance[new_track_id][object_type] = {
                            'box': item['box'],
                            'frame_id': frame_id,
                            'confidence': confidence
                        }
    
                        if object_type == 'head':
                            if new_track_id not in head_to_face_mapping:
                                head_to_face_mapping[new_track_id] = {}
                            
                            for face_item in objects.get('face', []):
                                if face_item['new_track_id'] == new_track_id:
                                    head_to_face_mapping[new_track_id] = {
                                        'face_box': face_item['box'],
                                        'face_frame_id': frame_id,
                                        'confidence': face_item['confidence']
                                    }
    
    return best_appearance, head_to_face_mapping

def process_single_json(json_path, video_path, output_path, video_predictor):
    """
    Hyperparam for Ground and Tracking
    """
    base_name = os.path.basename(video_path.replace(".mp4", ""))

    output_video_path = f"{output_path}/{base_name}"
    source_video_frame_dir = f"{output_path}/{base_name}/custom_video_frames"
    save_tracking_results_dir = f"{output_path}/{base_name}"
    save_tracking_mask_results_dir = f"{output_path}/{base_name}"
    save_corresponding_json_dir = f"{output_path}/{base_name}/corresponding_data.json"
    save_control_json_dir = f"{output_path}/{base_name}/control_sam2_frame.json"
    save_bbox_json_dir = f"{output_path}/{base_name}/valid_frame.json"

    if os.path.exists(save_bbox_json_dir) and os.path.exists(save_corresponding_json_dir) and os.path.exists(save_control_json_dir):
        print(f"Skipping {base_name}.")
        return

    with file_lock:
        os.makedirs(output_video_path, exist_ok=True)
        os.makedirs(source_video_frame_dir, exist_ok=True)
        os.makedirs(save_tracking_results_dir, exist_ok=True)
        os.makedirs(save_tracking_mask_results_dir, exist_ok=True)
        
    """
    Custom video input directly using video files
    """
    video_info = sv.VideoInfo.from_video_path(video_path)  # get video info
    frame_generator = sv.get_video_frames_generator(video_path, stride=1, start=0, end=None)

    # saving video to frames
    with file_lock:
        source_frames = Path(source_video_frame_dir)
        source_frames.mkdir(parents=True, exist_ok=True)

    total_frame_count = video_info.total_frames
    existing_frame_names = [
        p for p in os.listdir(source_video_frame_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    existing_frame_count = len(existing_frame_names)

    if source_frames.exists() and existing_frame_count == total_frame_count:
        print(f"Frames already exist in {source_video_frame_dir}, skipping.")
    else:
        with sv.ImageSink(
            target_dir_path=source_frames, 
            overwrite=True, 
            image_name_pattern="{:05d}.jpg"
        ) as sink:
            for frame in tqdm(frame_generator, desc="Saving Video Frames"):
                sink.save_image(frame)

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(source_video_frame_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # init video predictor state
    inference_state = video_predictor.init_state(video_path=source_video_frame_dir, async_loading_frames=True)

    with open(json_path, 'r') as json_file:
        bbox_json_data = json.load(json_file)

    max_people = estimate_num_people(bbox_json_data)

    best_bboxes, head_to_face_mapping = find_max_confidence_bbox(bbox_json_data, max_people)

    input_boxes = []
    OBJECT_IDS = []
    OBJECTS = []
    FRAME_IDX = []

    for new_track_id, object_data in best_bboxes.items():
        if 'face' in object_data:
            input_boxes.append([
                object_data['face']['box']['x1'],
                object_data['face']['box']['y1'],
                object_data['face']['box']['x2'],
                object_data['face']['box']['y2']
            ])
            OBJECT_IDS.append(new_track_id)
            OBJECTS.append('face')
            FRAME_IDX.append(int(object_data['face']['frame_id']))  # 记录face的frame_id
        
        if 'head' in object_data:
            input_boxes.append([
                object_data['head']['box']['x1'],
                object_data['head']['box']['y1'],
                object_data['head']['box']['x2'],
                object_data['head']['box']['y2']
            ])
            OBJECT_IDS.append(new_track_id)
            OBJECTS.append('head')
            FRAME_IDX.append(int(object_data['head']['frame_id']))  # 记录head的frame_id

        if 'person' in object_data:
            input_boxes.append([
                object_data['person']['box']['x1'],
                object_data['person']['box']['y1'],
                object_data['person']['box']['x2'],
                object_data['person']['box']['y2']
            ])
            OBJECT_IDS.append(new_track_id)
            OBJECTS.append('person')
            FRAME_IDX.append(int(object_data['person']['frame_id']))  # 记录person的frame_id

    input_boxes = np.array(input_boxes, dtype=np.float32)

    corresponding_json_data = {}
    control_json_data = {}

    Track_Object_IDs = OBJECT_IDS
    OBJECT_IDS = [i for i in range(1, 1 + len(input_boxes))]
    ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}

    for object_id, object_id_before, obj, frame_idx in zip(Track_Object_IDs, OBJECT_IDS, OBJECTS, FRAME_IDX):
        if object_id not in corresponding_json_data:
            corresponding_json_data[object_id] = {}
        corresponding_json_data[object_id][obj] = object_id_before

        if object_id not in control_json_data:
            control_json_data[object_id] = {}
        control_json_data[object_id][obj] = frame_idx

    with open(save_corresponding_json_dir, 'w') as f:
        json.dump(corresponding_json_data, f, indent=4)

    with open(save_control_json_dir, 'w') as f:
        json.dump(control_json_data, f, indent=4)

    """
    Step 3: Register each object's positive points to video predictor with separate add_new_points call
    """
    bbox_json_data = {}

    for object_id, track_id, object_name, box, frame_idx in zip(OBJECT_IDS, Track_Object_IDs, OBJECTS, input_boxes, FRAME_IDX):
        video_predictor.reset_state(inference_state)

        if track_id not in bbox_json_data:
            bbox_json_data[track_id] = {}

        if object_name == 'head':
            try:
                x_min, y_min, x_max, y_max = head_to_face_mapping[track_id]['face_box'].values()
                x_center_face = (x_min + x_max) / 2.0
                y_center_face = (y_min + y_max) / 2.0

                x_min, y_min, x_max, y_max = box
                x_center_head = (x_min + x_max) / 2.0
                y_center_head = (y_min + y_max) / 2.0

                points = np.array([[x_center_face, y_center_face], [x_center_head, y_center_head]], dtype=np.float32)
                labels = np.array([1, 1], np.int32)
            except:
                x_min, y_min, x_max, y_max = box
                x_center_head = (x_min + x_max) / 2.0
                y_center_head = (y_min + y_max) / 2.0

                points = np.array([[x_center_head, y_center_head]], dtype=np.float32)
                labels = np.array([1], np.int32)

            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=object_id,
                box=box,
                points=points,
                labels=labels,
            )
        else:
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=object_id,
                box=box,
            )


        """
        Step 4: Propagate the video predictor to get the segmentation results for each frame
        """
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, start_frame_idx=0):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        with file_lock:
            temp_save_tracking_mask_results_dir = os.path.join(save_tracking_mask_results_dir, "tracking_mask_results", str(object_id))
            if not os.path.exists(temp_save_tracking_mask_results_dir):
                os.makedirs(temp_save_tracking_mask_results_dir, exist_ok=True)

        """
        Step 5: Visualize the segment results across the video and save them
        """
        valid_frame_list = []
        for frame_idx, segments in video_segments.items():
            img = cv2.imread(os.path.join(source_video_frame_dir, frame_names[frame_idx]))
            
            object_ids = list(segments.keys())
            masks = list(segments.values())
            masks = np.concatenate(masks, axis=0)
            
            mask_img = torch.zeros(masks.shape[-2], masks.shape[-1])
            mask_img[masks[0] == True] = object_ids[0]
            mask_img = mask_img.numpy().astype(np.uint16)
            mask_img_pil = Image.fromarray(mask_img)
            mask_img_pil.save(os.path.join(temp_save_tracking_mask_results_dir, f"annotated_frame_{frame_idx:05d}.png"))

            if mask_img.max() != 0:
                valid_frame_list.append(frame_idx)

        if object_name not in bbox_json_data[track_id]:
            bbox_json_data[track_id][object_name] = []
        
        bbox_json_data[track_id][object_name].extend(valid_frame_list)

    with open(save_bbox_json_dir, 'w') as json_file:
        json.dump(bbox_json_data, json_file, indent=4)

    shutil.rmtree(source_video_frame_dir)

    torch.cuda.empty_cache()
    gc.collect()

def process_multi_files(json_video_pairs, output_path, sam2_checkpoint, model_cfg):
    print(f"Process {os.getpid()} is handling {len(json_video_pairs)} files.")
    
    if len(json_video_pairs) == 0:
        return
        
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    for json_file, video_file in tqdm(json_video_pairs, desc="Processing files"):
        process_single_json(json_file, video_file, output_path, video_predictor)
    
    print("Finish processing multiple files")

    del video_predictor
    torch.cuda.empty_cache()
    gc.collect()

    return True

def get_json_video_pairs(json_folder, video_folder):
    json_files = [os.path.join(json_folder, f) for f in os.listdir(json_folder) if f.endswith('.json')]
    video_files = {os.path.splitext(f)[0]: os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.mp4')}

    json_video_pairs = []
    for json_file in json_files:
        base_name = os.path.basename(json_file).replace(".json", "")
        if base_name in video_files:
            json_video_pairs.append((json_file, video_files[base_name]))

    return json_video_pairs

def main(json_folder, video_folder, output_path, sam2_checkpoint, model_cfg):
    json_video_pairs = get_json_video_pairs(json_folder, video_folder)
    process_multi_files(json_video_pairs, output_path, sam2_checkpoint, model_cfg)

if __name__ == "__main__":
    args = parse_args()

    sam2_checkpoint = os.path.join(args.sam2_checkpoint_path, "sam2.1_hiera_large.pt")
    if not os.path.exists(sam2_checkpoint):
        print(f"Model not found, downloading from Hugging Face and Github...")
        hf_hub_download(repo_id="facebook/sam2.1-hiera-large", filename="sam2.1_hiera_large.pt", local_dir=args.sam2_checkpoint_path)
    else:
        print(f"Model already exists in {args.model_path}, skipping download.")
    
    main(args.json_folder, args.video_folder, args.output_path, sam2_checkpoint, args.model_cfg)