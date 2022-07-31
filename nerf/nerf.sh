THIS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

SUPPORTED_CMDS="prepare|train|render"

# thank you https://unix.stackexchange.com/a/111518
if [[ "$1" =~ ^($SUPPORTED_CMDS)$ ]]; then
    python $THIS_DIR/$1.py "${@:2}"
else
    echo "Usage: nerf <cmd> <options>"
    echo "Where <cmd> is one of: $SUPPORTED_CMDS"
fi
