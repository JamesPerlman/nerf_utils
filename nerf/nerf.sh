THIS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

CONDA_CMDS="prepare|train|render|reorient|preview"

OTHER_CMDS="rename_images|cull_frames"

# thank you https://unix.stackexchange.com/a/111518
if [[ "$1" =~ ^($CONDA_CMDS)$ ]]; then
    eval "$(conda shell.bash hook)"
    conda activate $NGP_CONDA_ENV
    python $THIS_DIR/$1.py "${@:2}"
    conda deactivate
elif [[ "$1" =~ ^($OTHER_CMDS)$ ]]; then
    python $THIS_DIR/$1.py "${@:2}"
else
    echo "Usage: nerf <cmd> <options>"
    echo "Where <cmd> is one of: $CONDA_CMDS|$OTHER_CMDS"
fi

