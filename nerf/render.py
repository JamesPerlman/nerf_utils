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
DEFAULT_NGP_SCALE = 1.0 # 0.33
DEFAULT_NGP_ORIGIN = np.array([0.0, 0.0, 0.0]) # np.array([0.5, 0.5, 0.5])

# convenience method to parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Render script")

    parser.add_argument("--gpus", type=str, default="all", help="Which GPUs to use for rendering.  Example: \"0,1,2,3\" (Default: \"all\" = use all available GPUs)")
    parser.add_argument("--batch", type=str, default=None, help="For multi-GPU rendering. It is not recommended to use this feature directly.")

    parser.add_argument("--json", required=True, type=str, help="Path to a nerf-style transforms.json containing frames to render.")
    parser.add_argument("--frames_path", default="./rendered_frames", type=str, help="Path to a folder to save the rendered frames.")
    parser.add_argument("--overwrite", action="store_true", help="If enabled, images in the `--images_path` will be overwritten.  If not enabled, frames that already exist will not be re-rendered.")
    
    parser.add_argument("--samples_per_pixel", type=int, default=1, help="Number of samples per pixel.")

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

# deserializers

NGP_MASK_MODES = {
    "add": ngp.MaskMode.Add,
    "subtract": ngp.MaskMode.Subtract,
}

def deserialize_camera(camera: dict, render_data: dict) -> ngp.RenderCameraProperties:
    type = camera["type"]

    focal_length = 0.0
    w = render_data["w"]
    h = render_data["h"]
    cam_model = None

    cam_spher_quad = ngp.SphericalQuadrilateralConfig.Zero()
    cam_quad_hex = ngp.QuadrilateralHexahedronConfig.Zero()
    aperture_size = 0.0
    
    if type == "perspective":
        focal_length = camera["focal_len"]
        cam_model = ngp.CameraModel.Perspective
        aperture_size = camera["aperture"]
    elif type == "spherical_quadrilateral":
        cam_model = ngp.CameraModel.SphericalQuadrilateral
        # TODO: Figure out why we have to pass negative curvature here
        cam_spher_quad = ngp.SphericalQuadrilateralConfig(
            width=camera["sw"] * DEFAULT_NGP_SCALE,
            height=camera["sh"] * DEFAULT_NGP_SCALE,
            curvature=-camera["c"],
        )
    elif type == "quadrilateral_hexahedron":
        cam_model = ngp.CameraModel.QuadrilateralHexahedron

        [fsw, fsh] = DEFAULT_NGP_SCALE * np.array(camera["fs"])
        [bsw, bsh] = DEFAULT_NGP_SCALE * np.array(camera["bs"])
        sl = DEFAULT_NGP_SCALE * camera["sl"]
        cam_quad_hex = ngp.QuadrilateralHexahedronConfig(
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
    
    m = np.array(camera["m"])[:-1, :]

    return ngp.RenderCameraProperties(
        transform=m,
        model=cam_model,
        focal_length=focal_length,
        near_distance=camera["near"],
        aperture_size=aperture_size,
        focus_z=np.linalg.norm(camera["focus_target"] - m[:3, 3]),
        spherical_quadrilateral=cam_spher_quad,
        quadrilateral_hexahedron=cam_quad_hex,
    )


def deserialize_mask(mask: dict) -> list:
    if "mode" not in mask or "shape" not in mask:
        return None
    
    mode = NGP_MASK_MODES[mask["mode"]]
    transform = np.array(mask["transform"])
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

def deserialize_nerf(nerf: dict) -> ngp.NerfDescriptor:
    return ngp.NerfDescriptor(
        snapshot_path_str=nerf["path"],
        aabb=ngp.BoundingBox(np.array(nerf["aabb"]["min"]), np.array(nerf["aabb"]["max"])),
        transform=np.array(nerf["transform"]),
        modifiers=ngp.RenderModifiers(masks=[deserialize_mask(m) for m in nerf["modifiers"]["masks"]]),
        opacity=nerf["opacity"],
    )

def deserialize_render_request(render_data: dict, frame_index: int) -> ngp.RenderRequest:
    frame = render_data["frames"][frame_index]
    
    res = [render_data["w"], render_data["h"]]
    ds = ngp.DownsampleInfo.MakeFromMip(res, 0)
    output = ngp.RenderOutputProperties(
        res,
        ds,
        1, #spp
        ngp.ColorSpace.SRGB,
        ngp.TonemapCurve.Identity,
        0.0,
        np.array([0.0, 0.0, 0.0, 0.0]),
        False
    )

    camera = deserialize_camera(frame["camera"], render_data)

    modifiers = ngp.RenderModifiers(
        masks=[deserialize_mask(m) for m in frame["modifiers"]["masks"]],
    )
    nerfs = [deserialize_nerf(n) for n in frame["nerfs"]]

    aabb = ngp.BoundingBox(np.array([-8.0, -8.0, -8.0]), np.array([8.0, 8.0, 8.0]))

    return ngp.RenderRequest(
        output,
        camera,
        modifiers,
        nerfs,
        aabb
    )
        

# render images
def render_images(args: dict, render_data: dict):
    
    # initialize testbed
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    testbed.shall_train = False

    # global render props
    frame_width = int(render_data["w"])
    frame_height = int(render_data["h"])
    render_spp = args.samples_per_pixel

    # prepare frames directory
    Path(args.frames_path).mkdir(exist_ok=True, parents=True)
    rendered_frame_paths = []

    # render each frame via testbed
    # enumerate frames
    for idx, frame in enumerate(render_data["frames"]):
        # prepare output_path
        output_path = get_frame_output_path(args, frame)
        rendered_frame_paths.append(output_path)
        
        print(f"Rendering frame: {output_path}")

        # check if we can skip rendering this frame
        if not args.overwrite and output_path.exists():
            print(f"Frame already exists! Skipping...")
            continue

        # render the frame
        render_request = deserialize_render_request(render_data, idx)
        image = testbed.request_nerf_render_sync(render_request)

        # save frame as image
        Image.fromarray((image * 255).astype(np.uint8)).convert('RGBA').save(output_path.absolute())

# convenience method to fetch gpu indices via `nvidia-smi`
def get_gpus(args: dict) -> list[str]:
    proc = sp.Popen(['nvidia-smi', '--list-gpus'], stdout=sp.PIPE, stderr=sp.PIPE)
    out, err = proc.communicate()
    data = [line.decode() for line in out.splitlines(False)]
    gpus = [f"{item[4:item.index(':')]}" for item in data]
    if args.gpus and args.gpus != "all":
        gpus = [id for id in gpus if id in args.gpus]

    return gpus

# main
if __name__ == "__main__":
    args = parse_args()
    # load render json
    render_data = {}
    with open(args.json, 'r') as json_file:
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
