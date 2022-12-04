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
    parser.add_argument("--max_features", type=int, default=2<<12,
                        help="Maximum number of features to extract per image.")
    parser.add_argument("--max_matches", type=int, default=2<<13,
                        help="Maximum number of matches to extract per image pair.")
    parser.add_argument("--extract_only", action="store_true",
                        help="Only extract frames from video, do not run COLMAP.")

    return parser.parse_args()

# Thank you https://stackoverflow.com/a/23689767/892990
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

VIDEO_EXTENSIONS = [".mov", ".mp4", ".avi", ".mkv"]

# TODO: use Project Class
def parse_project(args) -> dict:
    project = dotdict({})

    project.path = Path(args.input)
    
    # Determine project.path and project.video_path
    if not project.path.is_dir():
        if project.path.suffix.lower() in VIDEO_EXTENSIONS:
            project.video_path = project.path
            project.path = project.path.parent
        else:
            raise Exception("Input must be a folder or a video file")
    
    project.images_path = project.path / "images"
    project.masks_path = project.path / "masks"
    project.colmap_db_path = project.path / "colmap.db"
    project.colmap_sparse_path = project.path / "colmap_sparse"
    project.colmap_text_path = project.path / "colmap_text"
    project.transforms_json_path = project.path / "transforms.json"
    project.done_path = project.path / "steps.done"
    project.metadata_dict_path = project.path / "metadata.json"

    project.use_masks = project.masks_path.exists()
        
    # check for images
    if not project.images_path.is_dir():
        def vid_path(name: str) -> Path:
            return project.path / name
        
        if project.video_path is None:
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

# extract frames step

def extract_frames(args):
    
    project = parse_project(args)

    if project.video_path is None:
        return

    # figure out if this step is done already
    extract_frames_done_path = project.done_path / "step.extract_frames.done"

    if extract_frames_done_path.exists():
        print("extract frames step is already done! Skipping...")
        return

    # TODO: Better conditions for deletion
    shutil.rmtree(project.images_path.absolute(), ignore_errors=True)

    project.images_path.mkdir(exist_ok=True, parents=True)

    def extract_frames_from_video(video_path: Path, start_frame: int = 0) -> list[Path]:
        cap = cv2.VideoCapture(project.video_path.as_posix())

        img_ext = "png"
        padding = 4
        
        def img_path(i: int) -> Path:
            filename = "{i:0{padding}d}.{img_ext}".format(i=i + start_frame, padding=padding, img_ext=img_ext)
            return f"./{Path(project.images_path / filename).as_posix()}"

        # Determine which frames to extract
        num_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if args.max_frames > num_video_frames:
            print(f"Error: You requested --max_frames={args.max_frames}, but this video only contains {num_video_frames} frames.  Using {num_video_frames} as the max.")
            args.max_frames = num_video_frames

        keep_indices = [int(i * (num_video_frames - 1) / (args.max_frames - 1)) for i in range(args.max_frames)]

        # TODO: filter by sharpness level
        # thank you https://vuamitom.github.io/2019/12/13/fast-iterate-through-video-frames.html
        success = cap.grab() # get the next frame       
        f = 0
        i = 0
        all_img_paths = []

        while success:
            if f in keep_indices:
                _, img = cap.retrieve()
                this_img_path = img_path(i)
                print(f"Extracting frame {i} of {args.max_frames} ({f} / {num_video_frames}): {this_img_path}...")
                cv2.imwrite(this_img_path, img)
                all_img_paths.append(this_img_path)
                i += 1
            
            # read next frame
            success = cap.grab()	
            f += 1

        return all_img_paths


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
            img_paths = extract_frames_from_video(video_path, i * args.max_frames)
            metadata_dict = {
                **metadata_dict,
                **make_metadata_dict(keys=img_paths, camera_id=i)
            }
            i += 1

    else:
        img_paths = extract_frames_from_video(project.video_path)
        metadata_dict = make_metadata_dict(keys=img_paths, camera_id=0)
    
    with open(project.metadata_dict_path, 'w+') as f:
        json.dump(metadata_dict, f, indent=2)
    
    os.system(f"touch {path2str(extract_frames_done_path)}")


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
                {f'--ImageReader.mask_path {path2str(project.masks_path)}' if project.use_masks else ''} \
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
                {f'--SiftMatching.max_num_matches {args.max_matches}' if args.max_matches is not None else ''} \
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
    extract_frames(args)

    if args.extract_only:
        exit(0)
    
    run_colmap(args)

    save_transforms(args)

    print("All done!")
