#!/usr/bin/env bash

method="sea"
gpu=0
data="all"
split="all"
log_folder="../../output/DBPWY_log/"
mode='full'

while getopts "m:g:d:s:l:" opt;
do
    case ${opt} in
        m) method=$OPTARG ;;
        g) gpu=$OPTARG ;;
        d) data=$OPTARG ;;
        s) split=$OPTARG ;;
        l) log_folder=$OPTARG ;;
        o) mode=$OPTARG ;;
    esac
done

args_folder="args/"
echo "log folder: " ${log_folder}

training_data=('dbp_wd' 'dbp_yg')

data_splits=('0_3/')

py_code='main_from_args.py'
if [[ ${mode} != "full" ]]; then
    py_code=('main_from_args_wo_attr.py')
fi

for data_name in ${training_data[@]}; do
    echo ""
    echo ${data_name}
    log_folder_current=${log_folder}${method}/${data_name}/
    if [[ ! -d ${log_folder_current} ]];then
        mkdir -p ${log_folder_current}
        echo "create log folder: " ${log_folder_current}
    fi
    for data_split in ${data_splits[@]}; do
        args_file=${args_folder}${method}"_args_DWY100K.json"
        cur_time="`date +%Y%m%d%H%M%S`"
        log_div=${data_split//\//_}
        CUDA_VISIBLE_DEVICES=${gpu} python3 ${py_code} ${args_file} ${data_name} ${data_split} > ${log_folder_current}${method}_${data_name}_${log_div}${cur_time}
    done
done
