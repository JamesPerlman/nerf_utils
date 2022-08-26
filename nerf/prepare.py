import json
import math
import os
from pathlib import Path
import shutil
import sys
import numpy as np
import cv2

from argparse import ArgumentParser

# get ffmpeg utils
sys.path.append(str(Path(__file__).parent / "../ffmpeg"))

from ffmpeg import get_frame_count


# Potentially, we can log out the commands we've already done
# if the command has already been run once before, then we can safely skip it
# The commands have to be in the right order

def os_system(cmd_str):
    # TODO: this should throw
    print(cmd_str)
    os.system(cmd_str)


def parse_args() -> dict:

    parser = ArgumentParser()

    parser.add_argument("--input", type=str, required=True,
                        help="A project folder, or a video path, or a folder containing an image sequence, or a folder containing a bunch of videos for batch processing.")
    parser.add_argument("--max_frames", type=int, default=250,
                        help="Maximum number of frames to extract, if given a video as input. Default=250, frames are extracted evenly over the duration of the video.")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="Directory for storing images")
    parser.add_argument("--matcher", default="sequential", choices=["exhaustive", "sequential"],
                        help="Which COLMAP matcher to use (sequential or exhaustive)")

    return parser.parse_args()

# Thank you https://stackoverflow.com/a/23689767/892990


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# TODO: use Project Class

def parse_project(args) -> dict:
    project = dotdict({})

    input_path = Path(args.input)

    # Determine project.path and project.video_path
    if input_path.is_dir():
        project.path = input_path
        project.video_path = project.path / "video.mov"
        # TODO: if not video.mov exists try video.mp4
    else:
        project.path = input_path.parent
        project.video_path = input_path
        # TODO: make sure input_video_path is an actual video

    # Determine images path

    if args.images_dir is None:
        # TODO: grab this from Project class
        project.images_path = project.path / "images"
    else:
        project.images_path = Path(args.images_dir)
    
    
    project.colmap_db_path = project.path / "colmap.db"

    project.colmap_sparse_path = project.path / "colmap_sparse"
    project.colmap_sparse_path.mkdir(exist_ok=True)

    project.colmap_text_path = project.path / "colmap_text"
    project.colmap_text_path.mkdir(exist_ok=True)

    project.transforms_json_path = project.path / "transforms.json"

    project.done_path = project.path / "steps.done"
    project.done_path.mkdir(exist_ok=True)

    return project


def path2str(path: Path) -> str:
    return f"\"{str(path.absolute())}\""

def qvec2rotmat(qvec):
    return np.array([
        [
            1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
        ],
        [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
        ],
        [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
        ]
    ])

def get_image_sharpness(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance_of_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance_of_laplacian

# FFMPEG step

def run_ffmpeg(args):
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error! --input path \"{input_path}\" does not exist.")
        exit()

    project = parse_project(args)

    # figure out if this step is done already
    ffmpeg_done_path = project.done_path / "step.ffmpeg.done"

    if ffmpeg_done_path.exists():
        print("ffmpeg step is already done! Skipping...")
        return

    # TODO: Better conditions for deletion
    shutil.rmtree(project.images_path.absolute(), ignore_errors=True)

    project.images_path.mkdir(exist_ok=True, parents=True)
    output_image_format = path2str(project.images_path / f"%04d.png")

    # Determine which frames to extract
    num_video_frames = get_frame_count(project.video_path)

    if args.max_frames > num_video_frames:
        print(
            f"Error: You requested --max_frames={args.max_frames}, but this video only contains {num_video_frames} frames.")
        exit()

    keep_indices = [int(i * (num_video_frames - 1) / (args.max_frames - 1))
                        for i in range(args.max_frames)]
    ffmpeg_select_str = "+".join([f"eq(n\,{i})" for i in keep_indices])

    # TODO: filter by sharpness level

    # Run ffmpeg
    os_system(f"\
        ffmpeg \
            -i {path2str(project.video_path)} \
            -qscale:v 1 \
            -qmin 1 \
            -vf \"\
                mpdecimate, \
                setpts=N/FRAME_RATE/TB, \
                select='{ffmpeg_select_str}'\
            \" \
            -vsync vfr \
            {output_image_format} \
    ")

    os_system(f"touch {path2str(ffmpeg_done_path)}")


def run_colmap(args):

    project = parse_project(args)

    # TODO: Use Project class

    colmap_feature_extractor_done_path = project.done_path / \
        "step.colmap_feature_extractor.done"
    if not colmap_feature_extractor_done_path.exists():
        os_system(f"\
            colmap feature_extractor \
                --ImageReader.camera_model OPENCV \
                --ImageReader.single_camera 1 \
                --SiftExtraction.estimate_affine_shape=true \
                --SiftExtraction.domain_size_pooling=true \
                --database_path {path2str(project.colmap_db_path)} \
                --image_path {path2str(project.images_path)} \
        ")

        os_system(f"touch {path2str(colmap_feature_extractor_done_path)}")

    colmap_matcher_done_path = project.done_path / "step.colmap_matcher.done"
    if not colmap_matcher_done_path.exists():
        os_system(f"\
            colmap {args.matcher}_matcher \
                --SiftMatching.guided_matching=true \
                --database_path {path2str(project.colmap_db_path)} \
        ")

        os_system(f"touch {path2str(colmap_matcher_done_path)}")

    colmap_mapper_done_path = project.done_path / "step.colmap_mapper.done"
    if not colmap_mapper_done_path.exists():
        # TODO: research --Mapper.ba_global_use_pba
        os_system(f"\
            colmap mapper \
                --database_path {path2str(project.colmap_db_path)} \
                --image_path {path2str(project.images_path)} \
                --output_path {path2str(project.colmap_sparse_path)} \
        ")

        os_system(f"touch {path2str(colmap_mapper_done_path)}")

    colmap_bundle_adjuster_done_path = project.done_path / "step.colmap_bundle_adjuster.done"
    if not colmap_bundle_adjuster_done_path.exists():
        os_system(f"\
            colmap bundle_adjuster \
                --input_path {path2str(project.colmap_sparse_path / '0')} \
                --output_path {path2str(project.colmap_sparse_path / '0')} \
                --BundleAdjustment.refine_principal_point 1 \
        ")

        os_system(f"touch {path2str(colmap_bundle_adjuster_done_path)}")

    colmap_model_orientation_aligner_done_path = project.done_path / \
        "step.colmap_model_orientation_aligner.done"
    if not colmap_model_orientation_aligner_done_path.exists():
        os_system(f"\
            colmap model_orientation_aligner \
                --image_path {path2str(project.images_path)} \
                --input_path {path2str(project.colmap_sparse_path / '0')} \
                --output_path {path2str(project.colmap_sparse_path / '0')} \
                --method IMAGE-ORIENTATION \
        ")
        os_system(
            f"touch {path2str(colmap_model_orientation_aligner_done_path)}")

    # TODO: Skip if already done
    colmap_model_converter_done_path = project.done_path / "step.colmap_model_converter.done"
    if not colmap_model_converter_done_path.exists():
        os_system(f"\
            colmap model_converter \
                --input_path {path2str(project.colmap_sparse_path / '0')} \
                --output_path {path2str(project.colmap_text_path)} \
                --output_type TXT \
        ")
        os_system(f"touch {path2str(colmap_model_converter_done_path)}")


def save_transforms(args):

    project = parse_project(args)
    
    if project.transforms_json_path.exists():
        print(f"{project.transforms_json_path} already exists! Skipping...")
        return

    print("Saving transforms...")

    cam = dotdict({})

    with open(project.colmap_text_path / "cameras.txt", "r") as text:
        for line in text:
            if line[0] == "#":
                continue
            field = line.split(" ")
            cam.w = float(field[2])
            cam.h = float(field[3])
            cam.fl_x = float(field[4])
            cam.fl_y = float(field[4])
            cam.k1 = k2 = p1 = p2 = 0
            cam.cx = cam.w / 2
            cam.cy = cam.h / 2
            cam_type = field[1]
            if cam_type == "OPENCV":
                cam.fl_y = float(field[5])
                cam.cx = float(field[6])
                cam.cy = float(field[7])
                cam.k1 = float(field[8])
                cam.k2 = float(field[9])
                cam.p1 = float(field[10])
                cam.p2 = float(field[11])
            else:
                print(f"Unhandled camera type \"{cam_type}\"")
            
            cam.camera_angle_x = math.atan(cam.w / (cam.fl_x * 2)) * 2
            cam.camera_angle_y = math.atan(cam.h / (cam.fl_y * 2)) * 2


    # Prepare to write frames json
    m3 = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
    flip_mat = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    
    with open(project.colmap_text_path / "images.txt", "r") as f:
        i = 0
        out = {
            **cam,
            "aabb_scale": 16,
            "frames": [],
        }

        for line in f:
            line = line.strip()
            if line[0] == "#":
                continue
            i = i + 1
            
            if  i % 2 == 1:
                fields = line.split(" ")
                img_path_str = f"./{project.images_path.name}/{fields[9]}"

                qvec = np.array(tuple(map(float, fields[1:5])))
                tvec = np.array(tuple(map(float, fields[5:8])))
                R = qvec2rotmat(-qvec)
                t = tvec.reshape([3,1])
                m = np.concatenate([np.concatenate([R, t], 1), m3], 0)
                c2w = np.matmul(np.linalg.inv(m), flip_mat)

                out["frames"].append({
                    "file_path" : img_path_str,
                    # "sharpness" : get_image_sharpness(image_path_str),
                    "transform_matrix" : c2w.tolist(),
                    "orientation": c2w[:3,:3].tolist(),
                    "translation": c2w[:3,-1].tolist(),
                })

                print(f"Using image {img_path_str}...")
    
    num_frames = len(out["frames"])

    print(f"Sorting transforms by image names...")
    out["frames"] = sorted(out["frames"], key=lambda f: int(Path(f["file_path"]).stem))

    print(f"Writing transforms ({num_frames} frames) to: {project.transforms_json_path.absolute()}")
    with open(str(project.transforms_json_path.absolute()), "w+") as outfile:
        json.dump(out, outfile, indent=2)

if __name__ == "__main__":
    args = parse_args()

    run_ffmpeg(args)

    run_colmap(args)

    save_transforms(args)

    print("All done!")
