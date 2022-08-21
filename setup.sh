#!/bin/sh
THIS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

alias extract_frames="python $THIS_DIR/ffmpeg/extract_frames.py"

alias nerf="$THIS_DIR/nerf/nerf.sh"

export NGP_DIR="/c/Users/bizon/Developer/instant-ngp"
export NGP_CONDA_ENV="instant-ngp"
