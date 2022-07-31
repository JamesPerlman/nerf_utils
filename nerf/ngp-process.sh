THIS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_DIR=$(echo "$1" | sed 's:/*$::')
DONE_TXT="${BASE_DIR}/done.txt"
NGP_DIR="/c/Users/bizon/Developer/instant-ngp"

if [ -f "${DONE_TXT}" ]; then
    exit 0
fi

cd $BASE_DIR \
    && python $NGP_DIR/scripts/colmap2nerf.py \
        --skip_prompts \
        --run_colmap \
        --colmap_matcher sequential \
        --colmap_db "${BASE_DIR}/colmap.db" \
        --keep_colmap_coords \
        --images "${BASE_DIR}/images" \
        --text "${BASE_DIR}/colmap_text" \
        --video_in "${BASE_DIR}/video.mp4" \
        --video_fps 5 \
        --camera_params "2184,2184,540,960,0,0,0,0" \
        --out "${BASE_DIR}/transforms.json" \
    && touch "${BASE_DIR}/done.txt"

