from argparse import ArgumentParser
from pathlib import Path
from ffmpeg import extract_frames

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, required=True, help="Path to output folder where frames will be generated")
    parser.add_argument("--skip", type=int, default=0, help="Only render out ever n frames")
    parser.add_argument("--zeroes", type=int, default=6, help="Leading zeroes to prepend to frame names")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"--input path: ${input_path} does not exist! Cannot extract frames.")
        exit()
    
    output_path.mkdir(exist_ok=True)

    extract_frames(
        input_video_path=input_path,
        output_frames_path=output_path,
        skip_frames=args.skip,
        leading_zeroes=args.zeroes
    )
