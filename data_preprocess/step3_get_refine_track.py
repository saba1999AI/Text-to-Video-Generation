import os
import json
import argparse
from tqdm import tqdm
from itertools import count
from concurrent.futures import ThreadPoolExecutor


def parse_args():
    parser = argparse.ArgumentParser(description="Video Processing Parameters")
    parser.add_argument('--input_json_folder', type=str, default='step1/json', help='Folder containing input JSON files.')
    parser.add_argument('--output_json_folder', type=str, default='step1/refine_bbox_jsons', help='Folder to save output JSON files.')
    parser.add_argument('--error_log_path', type=str, default='error_log.txt', help='Path to save error logs.')
    parser.add_argument('--num_processes', type=int, default=1, help='Maximum number of threads for concurrent processing.')
    args = parser.parse_args()
    return args


def compute_iou(box1, box2):
    x1 = max(box1['x1'], box2['x1'])
    y1 = max(box1['y1'], box2['y1'])
    x2 = min(box1['x2'], box2['x2'])
    y2 = min(box1['y2'], box2['y2'])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])

    union = area1 + area2 - intersection
    return intersection / union if union != 0 else 0


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


def find_frames_with_fewer_person_boxes(data_bbox, max_people):
    frames_with_fewer_person_boxes = []
    for frame, objects in data_bbox.items():
        num_persons_bbox = len(objects.get("person", []))
        num_persons_pose = len(data_bbox.get(frame, [])["pose"])
        if num_persons_bbox < max_people and num_persons_pose > num_persons_bbox:
            frames_with_fewer_person_boxes.append(frame)
    return frames_with_fewer_person_boxes


def fix_bbox_with_pose(json_bbox, frames):
    for frame in frames:
        if len(json_bbox[str(frame)]['pose']) != 0:
            pose_persons = json_bbox[str(frame)]['pose']
            json_bbox[str(frame)]["person"] = [{"name": "person",
                                           "class": person["class"],
                                           "confidence": person["confidence"],
                                           "box": person["box"],
                                           "track_id": person.get("track_id")}
                                          for person in pose_persons]
    return json_bbox



def assign_track_ids(data, cus_iou=0.7):
    max_people = estimate_num_people(data)

    if max_people == 0:
        raise ValueError("The maximum number of people cannot be determined, the data may be wrong")

    global_track_ids = count(1)
    last_frame_tracks = {}

    for frame_id, objects in data.items():
        current_frame_tracks = {}
        available_track_ids = {track_id: None for track_id in last_frame_tracks.keys()}

        persons = objects.get('person', [])
        heads = objects.get('head', [])
        faces = objects.get('face', [])

        used_heads = set()
        used_faces = set()

        for person in persons:
            person_box = person['box']
            best_iou = 0
            best_track_id = None

            for prev_track_id, (prev_face_box, prev_head_box, prev_person_box) in last_frame_tracks.items():
                iou = compute_iou(person_box, prev_person_box)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = prev_track_id

            if best_track_id is not None and best_iou > cus_iou:
                track_id = best_track_id
                available_track_ids.pop(track_id, None)
            else:
                track_id = next(global_track_ids)

            person['new_track_id'] = track_id
            current_frame_tracks[track_id] = (None, None, person_box)

            best_head_iou = 0
            best_head_index = None
            for head_index, head in enumerate(heads):
                if head_index in used_heads:
                    continue
                head_box = head['box']
                iou = compute_iou(person_box, head_box)
                if iou > cus_iou and iou > best_head_iou:
                    best_head_iou = iou
                    best_head_index = head_index

            if best_head_index is not None:
                head_box = heads[best_head_index]['box']
                heads[best_head_index]['new_track_id'] = track_id
                used_heads.add(best_head_index)
                current_frame_tracks[track_id] = (None, head_box, person_box)

            best_face_iou = 0
            best_face_index = None
            for face_index, face in enumerate(faces):
                if face_index in used_faces:
                    continue
                face_box = face['box']
                iou = compute_iou(person_box, face_box)
                if iou > cus_iou and iou > best_face_iou:
                    best_face_iou = iou
                    best_face_index = face_index

            if best_face_index is not None:
                face_box = faces[best_face_index]['box']
                faces[best_face_index]['new_track_id'] = track_id
                used_faces.add(best_face_index)
                current_frame_tracks[track_id] = (face_box, current_frame_tracks[track_id][1], person_box)

        for head_index, head in enumerate(heads):
            if head_index not in used_heads:
                best_person_iou = 0
                best_person_id = None
                for track_id, (_, _, person_box) in current_frame_tracks.items():
                    iou = compute_iou(head['box'], person_box)
                    if iou > best_person_iou:
                        best_person_iou = iou
                        best_person_id = track_id
                if best_person_id is not None:
                    head['new_track_id'] = best_person_id

        for face_index, face in enumerate(faces):
            if face_index not in used_faces:
                best_person_iou = 0
                best_person_id = None
                for track_id, (_, _, person_box) in current_frame_tracks.items():
                    iou = compute_iou(face['box'], person_box)
                    if iou > best_person_iou:
                        best_person_iou = iou
                        best_person_id = track_id
                if best_person_id is not None:
                    face['new_track_id'] = best_person_id

        last_frame_tracks = current_frame_tracks

    refined_data = refine(data, max_people)
    return refined_data if refined_data else data


def refine(data, max_people):
    def find_matching_frame(frame_id, bbox_count, direction=1):
        current_frame_id = int(frame_id) + direction
        while 0 <= current_frame_id < len(data):
            persons = data[str(current_frame_id)].get('person', [])
            if len(persons) == bbox_count:
                return current_frame_id
            current_frame_id += direction
        return None

    for frame_id, objects in data.items():
        persons = objects.get('person', [])
        faces = objects.get('face', [])

        if len(persons) == 0 and len(faces) > 0:
            matching_frame_id = find_matching_frame(frame_id, len(faces))
            if matching_frame_id is not None:
                matching_persons = data[str(matching_frame_id)]['person']
                for face, person in zip(faces, matching_persons):
                    face['new_track_id'] = person['new_track_id']

    for frame_id, objects in data.items():
        persons = objects.get('person', [])

        if len(persons) > 0:
            track_ids = [p['new_track_id'] for p in persons]
            if max(track_ids) > max_people:
                matching_frame_id = find_matching_frame(frame_id, len(persons))
                if matching_frame_id is not None:
                    matching_persons = data[str(matching_frame_id)]['person']
                    id_mapping = {p['new_track_id']: mp['new_track_id'] for p, mp in zip(persons, matching_persons)}
                    for person in persons:
                        person['new_track_id'] = id_mapping[person['new_track_id']]

    return data if data else None


def process_file(filename, input_json_folder, output_json_folder, error_log_path):
    input_path = os.path.join(input_json_folder, filename)
    output_path = os.path.join(output_json_folder, filename)

    if os.path.exists(output_path):
        print(f"Skipping {output_path}, as output files already exist.")
        return
        
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # refine json
        max_people = estimate_num_people(data)
        frames = find_frames_with_fewer_person_boxes(data, max_people)
        if max_people == 0:
            frames = list(range(0, len(data)))
        data_refine = fix_bbox_with_pose(data, frames)

        # refine track id
        data_with_new_ids = assign_track_ids(data_refine)
        
        with open(output_path, 'w') as f:
            json.dump(data_with_new_ids, f, indent=4)

    except ValueError as e:
        if str(e) == "The maximum number of people cannot be determined, the data may be wrong":
            with open(error_log_path, 'a') as error_file:
                error_file.write(filename + '\n')
            print(f"Error processing {filename}: {e}")


def main():
    args = parse_args()

    os.makedirs(args.output_json_folder, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.num_processes) as executor:
        for filename in tqdm(os.listdir(args.input_json_folder)):
            if filename.endswith(".json"):
                executor.submit(process_file, filename, args.input_json_folder, args.output_json_folder, args.error_log_path)


if __name__ == "__main__":
    main()