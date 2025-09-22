#!/bin/bash

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

PATH_TO_OFFICIAL_TNT="data/TnT"
PATH_TO_PREPROC_TNT="data/TnT"

if [ $# -lt 3 ]; then
    echo "Usage: $0 <output_dir> <max_subsets> <selected_subset> <other_args...>"
    echo "Example: $0 results 3 1 --exp_args"
    exit 1
fi

output_dir=$1
max_subsets=$2
selected_subset=$3
shift 3

all_scenes=("Barn" "Caterpillar" "Courthouse" "Meetingroom" "Ignatius" "Truck")
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
    local scene_name="$1"
    shift
    local output_dir="$1"
    shift
    local exp_args="$*"
    if [ "$scene_name" == "Courthouse" ]; then
        python train.py --cfg_files cfg/tnt_mesh.yaml --source_path $PATH_TO_PREPROC_TNT/$scene_name/ --model_path $output_dir/$scene_name $exp_args --depthanythingv2_overall --lambda_ascending 0.01 --lambda_T_inside 0 --white_background --depthanythingv2_alpha_adjust
    else
        python train.py --cfg_files cfg/tnt_mesh.yaml --source_path $PATH_TO_PREPROC_TNT/$scene_name/ --model_path $output_dir/$scene_name $exp_args
    fi
    python render.py $output_dir/$scene_name --skip_test --eval_fps
    python render.py $output_dir/$scene_name --skip_test --use_jpg
    python render_fly_through.py $output_dir/$scene_name/
    # python extract_mesh.py $output_dir/$scene_name/ --save_gpu --use_vert_color --final_lv 11 --adaptive --mesh_fname mesh_svr
    
    if [ "$scene_name" == "Caterpillar" ] || [ "$scene_name" == "Truck" ]; then
        PYTHONPATH=./ python mesh_extract/tsdf_mesh_tnt.py $output_dir/$scene_name --num_cluster 1 --use_depth_filter --sdf_trunc_scale 1.0
    else
        PYTHONPATH=./ python mesh_extract/tsdf_mesh_tnt.py $output_dir/$scene_name --num_cluster 1 --use_depth_filter --sdf_trunc_scale 4.0
    fi    
    
    /<your_path>/envs/open3d090/bin/python scripts/tnt_eval_vanilla/run.py \
        --dataset-dir $PATH_TO_OFFICIAL_TNT/$scene_name/ \
        --traj-path $PATH_TO_PREPROC_TNT/$scene_name/"$scene_name"_COLMAP_SfM.log \
        --ply-path $output_dir/$scene_name/mesh/tsdf/tsdf_fusion_post.ply
}

ulimit -n 2048  # Increase maximum number of files the script can read

for scene in "${selected_scenes[@]}"; do
    echo "============ start $scene ============"
    lanuch_exp "$scene" "$output_dir" "$@"
    echo "============ end $scene ============"
done