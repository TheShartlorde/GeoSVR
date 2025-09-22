#!/bin/bash

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

DATA_ROOT=data/360_v2

if [ $# -lt 3 ]; then
    echo "Usage: $0 <output_dir> <max_subsets> <selected_subset> <other_args...>"
    echo "Example: $0 results 3 1 --exp_args"
    exit 1
fi

output_dir=$1
max_subsets=$2
selected_subset=$3
shift 3

scenes_with_params=(
    "bonsai images_2"
    "counter images_2"
    "kitchen images_2"
    "room images_2"
    "bicycle images_4"
    "garden images_4"
    "stump images_4"
    "treehill images_4"
    "flowers images_4"
)

total_scenes=${#scenes_with_params[@]}

scenes_per_subset=$(( (total_scenes + max_subsets - 1) / max_subsets ))

start_index=$(( (selected_subset - 1) * scenes_per_subset ))
end_index=$(( start_index + scenes_per_subset - 1 ))

if [ $end_index -ge $total_scenes ]; then
    end_index=$(( total_scenes - 1 ))
fi

selected_scenes=("${scenes_with_params[@]:$start_index:$((end_index - start_index + 1))}")

echo "============ Configuration ============"
echo "Total scenes: $total_scenes"
echo "Max subsets: $max_subsets"
echo "Selected subset: $selected_subset"
echo "Scenes per subset: $scenes_per_subset"
echo "Running scenes:"
printf '%s\n' "${selected_scenes[@]}"
echo "Additional args: $@"
echo "======================================="

lanuch_exp() {
    local scene_name="$1"
    local output_dir="$2"
    shift 2
    local exp_args=("$@")

    echo "Starting experiment for $scene_name with params: ${exp_args[*]}"
    
    python train.py --cfg_files cfg/mipnerf360_mesh.yaml \
        --source_path $DATA_ROOT/$scene_name/ \
        --model_path $output_dir/$scene_name \
        "${exp_args[@]}"

    python render.py $output_dir/$scene_name --skip_train --eval_fps
    python render.py $output_dir/$scene_name
    python eval.py $output_dir/$scene_name/
    python render_fly_through.py $output_dir/$scene_name
    PYTHONPATH=./ python mesh_extract/tsdf_mesh_360.py $output_dir/$scene_name/ --num_cluster 5
    python render_mesh.py $output_dir/$scene_name/

}

for scene_info in "${selected_scenes[@]}"; do
    scene_name=$(echo "$scene_info" | awk '{print $1}')
    images_param=$(echo "$scene_info" | awk '{print $2}')
    
    echo "============ start $scene_name ============"
    lanuch_exp "$scene_name" "$output_dir" --images "$images_param" "$@"
    echo "============ end $scene_name ============"
done