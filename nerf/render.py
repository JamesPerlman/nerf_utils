import argparse
import json
import math
import os
import re
import sys

import numpy as np
import subprocess as sp

from pathlib import Path
from PIL import Image

# pyngp
import load_ngp
import pyngp as ngp # noqa

# local imports
from project import get_project_snapshot_name

# constants
DEFAULT_NGP_SCALE = 0.33
DEFAULT_NGP_ORIGIN = np.array([0.5, 0.5, 0.5])

# convenience method to parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Render script")

    parser.add_argument("--gpus", type=str, default="all", help="Which GPUs to use for rendering.  Example: \"0,1,2,3\" (Default: \"all\" = use all available GPUs)")
    parser.add_argument("--batch", type=str, default=None, help="For multi-GPU rendering. It is not recommended to use this feature directly.")

    parser.add_argument("--transforms", type=str, help="Path to NeRF-style transforms.json.")
    parser.add_argument("--snapshots_path", type=str, help="Path to snapshots folder. Will be set automatically if not given.")
    parser.add_argument("--snapshot", type=str, help="Path to a single snapshot (*.msgpack) file.  If given, this will override any n_steps data inside transforms.json")

    parser.add_argument("--frames_json", required=True, type=str, help="Path to a nerf-style transforms.json containing frames to render.")
    parser.add_argument("--frames_path", required=True, type=str, help="Path to a folder to save the rendered frames.")
    parser.add_argument("--overwrite_frames", action="store_true", help="If enabled, images in the `--images_path` will be overwritten.  If not enabled, frames that already exist will not be re-rendered.")
    
    parser.add_argument("--samples_per_pixel", type=int, default=16, help="Number of samples per pixel.")

    parser.add_argument("--video_out", type=str, help="Path to a video to be exported. Uses ffmpeg to combine frames in order.")
    parser.add_argument("--video_fps", type=str, default="30", help="Use in combination with `--video_out`. Sets the fps of the output video.")

    return parser.parse_args()

# escapes a string for use in ffmpeg's playlist.txt
def safe_str(string_like) -> str:
    return re.sub(r'([\" \'])', r'\\\1', str(string_like))

# convenience function to generate a consistent frame output path
def get_frame_output_path(args: dict, frame: dict) -> Path:
    frame_path = Path(frame["file_path"])
    if frame_path.suffix == '':
        frame_path = Path(f"{frame_path}.png")

    return Path(args.frames_path) / frame_path.name

# convenience method to output a video from an image sequence, using ffmpeg
def export_video_sequence(args: dict, render_data: dict):
    if args.video_out == None:
        return
    
    # combine frames into a video via ffmpeg
    print("Rendering output via ffmpeg...") 

    frame_paths = [get_frame_output_path(args, frame) for frame in render_data["frames"]]
    frame_paths = [path for path in frame_paths if path.exists()]
    
    video_path = Path(args.video_out)
    video_path.unlink(missing_ok=True)
    fps = args.video_fps

    # fetch all images and save to a playlist
    playlist_path = video_path.parent / f"{video_path.stem}-playlist.txt"
    playlist_path.unlink(missing_ok=True)
    print(playlist_path)

    # prepare ffmpeg playlist.txt, each line is `file 'path/to/image'`
    ffmpeg_files = [f"file '{safe_str(p.absolute())}'" for p in frame_paths]
    playlist_str = "\n".join(ffmpeg_files)

    with open(playlist_path, "w+") as f:
        f.write(playlist_str)
    
    os.system(f"\
        ffmpeg \
            -f concat \
            -safe 0 \
            -r {fps} \
            -i \"{playlist_path}\" \
            -c:v libx264 \
            -pix_fmt yuv420p \
            -vf fps={fps} \
            \"{video_path}\" \
        ")
    
    playlist_path.unlink(missing_ok=True)

def generate_snapshots(args: dict, render_data: dict):
    
    if args.snapshot != None:
        return
    
    print("Generating training snapshots...")

    # figure out which steps we have to train
    training_steps = []
    if "n_steps" in render_data:
        training_steps = [render_data["n_steps"]]
    else:
        try:
            training_steps = [frame["n_steps"] for frame in render_data["frames"]]
        except:
            print("n_steps was not defined.  Either declare it at the root-level of transforms.json, or on a per-frame basis, or use the --snapshot argument.")
            return
    
    # now train
    str_steps = [str(step) for step in training_steps]
    train_py_path = Path(os.path.realpath(__file__)).parent / "train.py"
    snapshots_path = Path(args.snapshots_path).absolute()

    train_cmd = ["python", str(train_py_path.absolute()), "--transforms", str(Path(args.transforms).absolute()), "--save", str(snapshots_path), "--steps", *str_steps]
    train_proc = sp.Popen(train_cmd, env=os.environ, shell=True, stderr=sys.stderr, stdout=sys.stdout)
    train_proc.wait()

# ngp math utils

def nerf_matrix_to_ngp(nerf_matrix: np.matrix) -> np.matrix:
    result = nerf_matrix
    result[:, 0:3] *= DEFAULT_NGP_SCALE
    result[:3, 3] = result[:3, 3] * DEFAULT_NGP_SCALE + DEFAULT_NGP_ORIGIN.reshape(3, 1)

    # Cycle axes xyz<-yzx
    result[:3, :] = np.roll(result[:3, :], -1, axis=0)

    return result


# convenience method to convert NeRF point to NGP
def nerf2ngp(
        xyz: np.array,
        origin = DEFAULT_NGP_ORIGIN,
        scale = DEFAULT_NGP_SCALE
    ) -> np.array:
    xyz_cycled = np.array([xyz[1], xyz[2], xyz[0]])
    return scale * xyz_cycled + origin


# deserializers

NGP_MASK_MODES = {
    "add": ngp.MaskMode.Add,
    "subtract": ngp.MaskMode.Subtract,
}

def deserialize_mask(mask: dict) -> list:
    if "mode" not in mask or "shape" not in mask:
        return None
    
    mode = NGP_MASK_MODES[mask["mode"]]
    transform = nerf_matrix_to_ngp(np.matrix(mask["transform"]))
    opacity = mask["opacity"]
    feather = mask["feather"]
    shape = mask["shape"]

    if shape == "box":
        dims = np.array(mask["dims"])
        return ngp.Mask3D.Box(dims, transform, mode, feather, opacity)
    elif shape == "cylinder":
        r = mask["radius"]
        h = mask["height"]
        return ngp.Mask3D.Cylinder(r, h, transform, mode, feather, opacity)
    elif shape == "sphere":
        r = mask["radius"]
        return ngp.Mask3D.Sphere(r, transform, mode, feather, opacity)

    print(f"Warning: Unknown mask shape detected: {shape}")
    return None

# render images
def render_images(args: dict, render_data: dict):
    # initialize testbed
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    testbed.shall_train = False
    
    if args.snapshot:
        testbed.load_snapshot(args.snapshot)
    
    testbed.fov_axis = 0
    testbed.background_color = [0.0, 0.0, 0.0, 0.0]


    # global render props
    frame_width = int(render_data["w"])
    frame_height = int(render_data["h"])
    render_spp = args.samples_per_pixel

    # prepare frames directory
    Path(args.frames_path).mkdir(exist_ok=True, parents=True)
    rendered_frame_paths = []

    # render each frame via testbed
    for frame in render_data["frames"]:
        # prepare output_path
        output_path = get_frame_output_path(args, frame)
        rendered_frame_paths.append(output_path)
        
        print(f"Rendering frame: {output_path}")

        # check if we can skip rendering this frame
        if not args.overwrite_frames and output_path.exists():
            print(f"Frame already exists! Skipping...")
            continue
            
        if not args.snapshot and "n_steps" in frame:
            n_steps = frame["n_steps"]
            if int(n_steps) != testbed.training_step:
                snapshot_path = Path(args.snapshots_path) / get_project_snapshot_name(n_steps)
                testbed.load_snapshot(str(snapshot_path.absolute()))

        # prepare testbed to render this frame
        
        if "aabb" in frame:
            aabb = frame["aabb"]
            aabb_min = np.array(aabb["min"])
            aabb_max = np.array(aabb["max"])

            testbed.render_aabb = ngp.BoundingBox(nerf2ngp(aabb_min), nerf2ngp(aabb_max))
        
        if "camera" in frame:
            camera = frame["camera"]
            type = camera["type"]

            if type == "perspective":
                testbed.render_camera_model = ngp.CameraModel.Perspective
                testbed.fov = math.degrees(camera["fov"])
            elif type == "spherical_quadrilateral":
                testbed.render_camera_model = ngp.CameraModel.SphericalQuadrilateral
                # TODO: Figure out why we have to pass negative curvature here
                testbed.camera_spherical_quadrilateral = ngp.SphericalQuadrilateralConfig(
                    width=camera["sw"] * DEFAULT_NGP_SCALE,
                    height=camera["sh"] * DEFAULT_NGP_SCALE,
                    curvature=-camera["c"],
                )
            elif type == "quadrilateral_hexahedron":
                testbed.render_camera_model = ngp.CameraModel.QuadrilateralHexahedron

                [fsw, fsh] = DEFAULT_NGP_SCALE * np.array(camera["fs"])
                [bsw, bsh] = DEFAULT_NGP_SCALE * np.array(camera["bs"])
                sl = DEFAULT_NGP_SCALE * camera["sl"]
                testbed.camera_quadrilateral_hexahedron = ngp.QuadrilateralHexahedronConfig(
                    front=ngp.Quadrilateral3D(
                        tl=np.array(0.5 * np.array([-fsw, -fsh, sl])),
                        tr=np.array(0.5 * np.array([fsw, -fsh, sl])),
                        bl=np.array(0.5 * np.array([-fsw, fsh, sl])),
                        br=np.array(0.5 * np.array([fsw, fsh, sl])),
                    ), 
                    back=ngp.Quadrilateral3D(
                        tl=np.array(0.5 * np.array([-bsw, -bsh, -sl])),
                        tr=np.array(0.5 * np.array([bsw, -bsh, -sl])),
                        bl=np.array(0.5 * np.array([-bsw, bsh, -sl])),
                        br=np.array(0.5 * np.array([bsw, bsh, -sl])),
                    ),
                )
            else:
                raise Exception(f"Unknown camera type: {type}")
            
        
            if "aperture" in camera:
                testbed.dof = camera["aperture"]
            
            if "focus_target" in camera:
                testbed.autofocus_target = nerf2ngp(np.array(camera["focus_target"]))
                testbed.autofocus = True
            
            testbed.render_near_distance = camera["near"] * DEFAULT_NGP_SCALE
            testbed.set_nerf_camera_matrix(np.matrix(camera["m"])[:-1,:])
        
        if "masks" in frame:
            testbed.render_masks = [deserialize_mask(mask) for mask in frame["masks"]]

        # render the frame
        image = testbed.render(frame_width, frame_height, render_spp, True)

        # save frame as image
        Image.fromarray((image * 255).astype(np.uint8)).convert('RGBA').save(output_path.absolute())

# convenience method to fetch gpu indices via `nvidia-smi`
def get_gpus(args: dict) -> list[str]:
    proc = sp.Popen(['nvidia-smi', '--list-gpus'], stdout=sp.PIPE, stderr=sp.PIPE)
    out, err = proc.communicate()
    data = [line.decode() for line in out.splitlines(False)]
    gpus = [f"{item[4:item.index(':')]}" for item in data]
    if args.gpus:
        gpus = [id for id in gpus if id in args.gpus]

    return gpus

# main
if __name__ == "__main__":
    args = parse_args()
    # load render json
    render_data = {}
    with open(args.frames_json, 'r') as json_file:
        render_data = json.load(json_file)

    if args.batch != None:
        print("Starting render on CUDA device: " + os.environ['CUDA_VISIBLE_DEVICES'])

        # only use a portion of the render_data["frames"]
        frames = render_data["frames"]
        [n, d] = [int(s) for s in args.batch.split('/')]
        t = len(frames)

        render_data["frames"] = [frames[i] for i in range(t) if i % d == n]
        render_images(args, render_data)
        
    # No --batch flag means we are part of the main process
    else:
        # First, generate snapshots
        generate_snapshots(args, render_data)

        # split into subprocesses, one for each gpu
        procs = []
        gpus = get_gpus(args)
        n_gpus = len(gpus)

        # In case there are less images to render than the number of gpus available...
        n_frames = len(render_data["frames"])
        if n_frames < n_gpus:
            gpus = gpus[0:n_frames]
            n_gpus = n_frames

        print(f"Using {n_gpus} GPU(s).  Rendering images...")
        
        i = 0
        for gpu in gpus:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            
            # rerun this command, but with a batch arg
            cmd = sys.argv.copy()
            cmd.insert(0, 'python')
            cmd.extend(["--batch", f"{i}/{n_gpus}"])

            proc = sp.Popen(cmd, env=env, shell=True, stderr=sys.stderr, stdout=sys.stdout)
            procs.append(proc)

            i = i + 1

        
        for p in procs:
            p.wait()

        export_video_sequence(args, render_data)
