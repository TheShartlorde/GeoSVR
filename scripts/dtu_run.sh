#!/bin/bash

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

DATA_ROOT=data/DTU_2dgs
PATH_TO_OFFICIAL_DTU="data/DTU"

if [ $# -lt 3 ]; then
    echo "Usage: $0 <output_dir> <max_subsets> <selected_subset> <other_args...>"
    echo "Example: $0 results 3 1 --exp_args"
    exit 1
fi

output_dir=$1
max_subsets=$2
selected_subset=$3
shift 3

all_scenes=(24 37 40 55 63 65 69 83 97 105 110 106 114 118 122)
total_scenes=${#all_scenes[@]}

scenes_per_subset=$(( (total_scenes + max_subsets - 1) / max_subsets ))

start_index=$(( (selected_subset - 1) * scenes_per_subset ))
end_index=$(( start_index + scenes_per_subset - 1 ))

if [ $end_index -ge $total_scenes ]; then
    end_index=$(( total_scenes - 1 ))
fi

selected_scenes=("${all_scenes[@]:$start_index:$((end_index - start_index + 1))}")

echo "============ Configuration ============"
echo "Total scenes: $total_scenes"
echo "Max subsets: $max_subsets"
echo "Selected subset: $selected_subset"
echo "Scenes per subset: $scenes_per_subset"
echo "Running scenes: ${selected_scenes[@]}"
echo "Additional args: $@"
echo "======================================="

lanuch_exp() {
    local scene_id="$1"
    shift
    local output_dir="$1"
    shift
    local exp_args="$*"

    local scene_name=scan"$scene_id"

    python train.py --cfg_files cfg/dtu_mesh.yaml --source_path $DATA_ROOT/"$scene_name"/ --model_path $output_dir/$scene_name $exp_args --test_iterations 6000 15000
    python render.py $output_dir/$scene_name --skip_test --eval_fps
    python render.py $output_dir/$scene_name --skip_test --use_jpg
    python render_fly_through.py $output_dir/$scene_name/

    if [ "$scene_name" == "scan110" ]; then
        PYTHONPATH=./ python mesh_extract/tsdf_mesh.py $output_dir/$scene_name/ --num_cluster 1 --voxel_size 0.002 --max_depth 5.0 --sdf_trunc_scale 1.5
    else
        PYTHONPATH=./ python mesh_extract/tsdf_mesh.py $output_dir/$scene_name/ --num_cluster 1 --voxel_size 0.002 --max_depth 5.0
    fi

    python scripts/eval_dtu_vanilla/evaluate_single_scene.py \
        --input_mesh $output_dir/$scene_name/mesh/tsdf/tsdf_fusion_post.ply \
        --scan_id $scene_id \
        --output_dir $output_dir/$scene_name/mesh/tsdf/ \
        --mask_dir $DATA_ROOT \
        --DTU $PATH_TO_OFFICIAL_DTU
}

ulimit -n 2048

for scene in "${selected_scenes[@]}"; do
    echo "============ start $scene ============"
    lanuch_exp "$scene" "$output_dir" "$@"
    echo "============ end $scene ============"
done