#!/bin/sh
THIS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

dng2png() {
    find $1 -name "*.dng" | parallel -I% --max-args 1 dcraw -c -w % '|' pnmtopng '>' $2/{/.}.png
}

alias nerf="$THIS_DIR/nerf/nerf.sh"

export NGP_DIR="/c/Users/bizon/Developer/blender-ngp"
export NGP_CONDA_ENV="blender-ngp"
