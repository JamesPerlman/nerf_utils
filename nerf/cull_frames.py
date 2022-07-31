from argparse import ArgumentParser
import json
from pathlib import Path


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to transforms.json")
    parser.add_argument("--frames", type=int, required=True, help="Final number of frames")
    parser.add_argument("--max", type=int, default=None, help="Max frame, filter out all frames above this number")
    parser.add_argument("--min", type=int, default=None, help="Min frame, filter out all frames below this number")
    parser.add_argument("--output", required=True, help="Path to output json file")

    return parser.parse_args()

def read_json(path: Path):
    with open(path, 'r') as f:
        return json.load(f)

def write_json(data: dict, path: Path):
    with open(path, 'w+') as f:
        json.dump(data, f, indent=4)

def frame_in_range(frame, min_frame, max_frame):
    file_path = Path(frame["file_path"])
    frame_number = int(file_path.stem)
    return frame_number >= min_frame and frame_number <= max_frame

if __name__ == "__main__":
    args = parse_args()
    out_frames = args.frames
    output_path = args.output
    input_path = Path(args.input)
    transforms_json = read_json(input_path)

    frames = transforms_json["frames"]
    num_frames = len(frames)

    max_frame = num_frames if args.max == None else min(num_frames, args.max)
    min_frame = 0 if args.min == None else max(0, args.min)

    keep_indices = [int(i * (num_frames - 1) / (out_frames - 1)) for i in range(out_frames)]

    transforms_json["frames"] = [frames[i] for i in keep_indices if frame_in_range(frames[i], min_frame, max_frame)]

    write_json(transforms_json, output_path)

    