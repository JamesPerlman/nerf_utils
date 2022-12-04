#!/bin/sh
THIS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

alias nerf="$THIS_DIR/nerf/nerf.sh"

export NGP_DIR="/c/Users/bizon/Developer/blender-ngp"
export NGP_CONDA_ENV="blender-ngp"
