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
                        help="Input folder for project")
    parser.add_argument("--max_frames", type=int, default=250,
                        help="Maximum number of frames to extract, if given a video as input. Default=250, frames are extracted evenly over the duration of the video.")
    parser.add_argument("--matcher", default="sequential", choices=["exhaustive", "sequential"],
                        help="Which COLMAP matcher to use (sequential or exhaustive)")
    parser.add_argument("--max_features", type=int, default=8192,
                        help="Maximum number of features to extract per image. Default=8192.")

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

    project.path = Path(args.input)
    project.images_path = project.path / "images"
    project.colmap_db_path = project.path / "colmap.db"
    project.colmap_sparse_path = project.path / "colmap_sparse"
    project.colmap_text_path = project.path / "colmap_text"
    project.transforms_json_path = project.path / "transforms.json"
    project.done_path = project.path / "steps.done"
    project.metadata_dict_path = project.path / "metadata.json"

    # Determine project.path and project.video_path
    if not project.path.is_dir():
        print(f"Input path must be a directory.")
        exit()
        
    # check for images
    if not project.images_path.is_dir():
        def vid_path(name: str) -> Path:
            return project.path / name
        
        project.video_path = vid_path("video.mov")
        if not project.video_path.exists():
            print(f"Video path \"{project.video_path}\" not found...")
            project.video_path = vid_path("video.mp4")
            print(f"Trying {project.video_path}...")
        if not project.video_path.exists():
            print(f"Video path \"{project.video_path}\" not found...")
            project.video_path = vid_path("videos")
            print(f"Trying {project.video_path}...")
        if not project.video_path.exists():
            print(f"Video path \"{project.video_path}\" not found...")
            print("Unable to find video source in this project.")

    # TODO: make sure input_video_path is an actual video

    project.colmap_sparse_path.mkdir(exist_ok=True)
    project.colmap_text_path.mkdir(exist_ok=True)
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
    
    project = parse_project(args)

    if project.video_path is None:
        return

    # figure out if this step is done already
    ffmpeg_done_path = project.done_path / "step.ffmpeg.done"

    if ffmpeg_done_path.exists():
        print("ffmpeg step is already done! Skipping...")
        return

    # TODO: Better conditions for deletion
    shutil.rmtree(project.images_path.absolute(), ignore_errors=True)

    project.images_path.mkdir(exist_ok=True, parents=True)

    def run_ffmpeg_on_video(video_path: Path, start_frame: int = 0) -> list[Path]:
        img_ext = "png"
        padding = 4
        output_image_format = path2str(project.images_path / f"%0{padding}d.{img_ext}")
        def img_path(i: int) -> Path:
            filename = "{i:0{padding}d}.png".format(i=i + start_frame, padding=padding)
            return f"./{Path(project.images_path / filename).as_posix()}"

        # Determine which frames to extract
        num_video_frames = get_frame_count(video_path)
        
        if args.max_frames > num_video_frames:
            print(f"Error: You requested --max_frames={args.max_frames}, but this video only contains {num_video_frames} frames.  Using {num_video_frames} as the max.")
            args.max_frames = num_video_frames

        keep_indices = [int(i * (num_video_frames - 1) / (args.max_frames - 1)) for i in range(args.max_frames)]
        ffmpeg_select_str = "+".join([f"eq(n\,{i})" for i in keep_indices])

        # TODO: filter by sharpness level

        # Run ffmpeg
        os_system(f"\
            ffmpeg \
                -i {path2str(video_path)} \
                -qscale:v 1 \
                -qmin 1 \
                -vf \"\
                    mpdecimate, \
                    setpts=N/FRAME_RATE/TB, \
                    select='{ffmpeg_select_str}'\
                \" \
                -start_number {start_frame} \
                -vsync vfr \
                {output_image_format} \
        ")

        return [img_path(i) for i in range(args.max_frames)]


    def make_metadata_dict(keys: list[Path], camera_id: int):
        return {
            str(keys[i]): {
                "camera_id": camera_id,
                "time_id": i,
            } for i in range(len(keys))
        }

    metadata_dict: dict = {}
    # if video_path is a directory, we need to run this on multiple files
    if project.video_path.is_dir():
        # TODO: Make sure that all these files are videos
        i = 0
        for video_path in project.video_path.iterdir():
            img_paths = run_ffmpeg_on_video(video_path, i * args.max_frames)
            metadata_dict = {
                **metadata_dict,
                **make_metadata_dict(keys=img_paths, camera_id=i)
            }
            i += 1

    else:
        img_paths = run_ffmpeg_on_video(project.video_path)
        metadata_dict = make_metadata_dict(keys=img_paths, camera_id=0)
    
    with open(project.metadata_dict_path, 'w+') as f:
        json.dump(metadata_dict, f, indent=2)
    
    os.system(f"touch {path2str(ffmpeg_done_path)}")


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
                --SiftExtraction.max_num_features={args.max_features} \
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
        # {args.matcher == 'sequential' and '--SequentialMatching.loop_detection=1' or ''} \

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
            cam.k1 = cam.k2 = cam.p1 = cam.p2 = 0
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

            break


    # Prepare to write frames json
    m3 = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
    flip_mat = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    metadata_dict: dict = None
    if project.metadata_dict_path.exists():
        with open(project.metadata_dict_path, 'r') as f:
            metadata_dict = json.load(f)
    
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
                
                if not Path(img_path_str).exists():
                    continue

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
                })

                print(f"Using image {img_path_str}...")
    
    num_frames = len(out["frames"])

    print(f"Sorting transforms by image names...")
    out["frames"] = sorted(out["frames"], key=lambda f: Path(f["file_path"]).stem)

    print(f"Writing transforms ({num_frames} frames) to: {project.transforms_json_path.absolute()}")
    with open(str(project.transforms_json_path.absolute()), "w+") as outfile:
        json.dump(out, outfile, indent=2)

if __name__ == "__main__":
    args = parse_args()

    run_ffmpeg(args)

    run_colmap(args)

    save_transforms(args)

    print("All done!")
