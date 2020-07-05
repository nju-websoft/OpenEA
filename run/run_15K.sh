#!/usr/bin/env bash

method="BootEA"
gpu=0
data="all"
split="all"
log_folder="../../output/log/"
mode='full'

while getopts "m:g:d:s:o:l:" opt;
do
    case ${opt} in
        m) method=$OPTARG ;;
        g) gpu=$OPTARG ;;
        d) data=$OPTARG ;;
        s) split=$OPTARG ;;
        o) mode=$OPTARG ;;
        l) log_folder=$OPTARG ;;
        *) echo "parameter error"
    esac
done

args_folder="args/"
echo "log folder: " "${log_folder}"

training_data=('EN_DE_15K_V1' 'EN_DE_15K_V2' 'EN_FR_15K_V1' 'EN_FR_15K_V2'
                 'D_W_15K_V1' 'D_W_15K_V2' 'D_Y_15K_V1' 'D_Y_15K_V2')
if [[ ${data} == "dev1" ]]; then
    training_data=('EN_DE_15K_V1')
elif [[ ${data} == "dev2" ]]; then
    training_data=('EN_DE_15K_V2')
elif [[ ${data} == "de" ]]; then
    training_data=('EN_DE_15K_V1' 'EN_DE_15K_V2')

elif [[ ${data} == "frv1" ]]; then
    training_data=('EN_FR_15K_V1')
elif [[ ${data} == "frv2" ]]; then
    training_data=('EN_FR_15K_V2')
elif [[ ${data} == "fr" ]]; then
    training_data=('EN_FR_15K_V1' 'EN_FR_15K_V2')

elif [[ ${data} == "wdv1" ]]; then
    training_data=('D_W_15K_V1')
elif [[ ${data} == "wdv2" ]]; then
    training_data=('D_W_15K_V2')
elif [[ ${data} == "wd" ]]; then
    training_data=('D_W_15K_V1' 'D_W_15K_V2')

elif [[ ${data} == "ygv1" ]]; then
    training_data=('D_Y_15K_V1')
elif [[ ${data} == "ygv2" ]]; then
    training_data=('D_Y_15K_V2')
elif [[ ${data} == "yg" ]]; then
    training_data=('D_Y_15K_V1' 'D_Y_15K_V2')
fi
echo "training data: " "${training_data[@]}"

data_splits=('721_5fold/1/' '721_5fold/2/' '721_5fold/3/' '721_5fold/4/' '721_5fold/5/')
if [[ ${split} == "1" ]]; then
    data_splits=('721_5fold/1/')
elif [[ ${split} == "2" ]]; then
    data_splits=('721_5fold/2/')
elif [[ ${split} == "3" ]]; then
    data_splits=('721_5fold/3/')
elif [[ ${split} == "4" ]]; then
    data_splits=('721_5fold/4/')
elif [[ ${split} == "5" ]]; then
    data_splits=('721_5fold/5/')
fi
echo "data splits: " "${data_splits[@]}"

py_code='main_from_args.py'
if [[ ${mode} = "wo_attr" ]]; then
    py_code='main_from_args_wo_attr.py'
elif [[ ${mode} = "test" ]]; then
    py_code='main_from_args_test.py'
    log_folder="../../output/test_log/"
elif [[ ${mode} = "rev" ]]; then
    py_code='main_from_args_reversed.py'
    log_folder="../../output/rev_log/"
fi
echo "py code: " "${py_code}"

for data_name in "${training_data[@]}"; do
    echo ""
    echo "${data_name}"
    log_folder_current=${log_folder}${method}/${data_name}/
    if [[ ! -d ${log_folder_current} ]];then
        mkdir -p "${log_folder_current}"
        echo "create log folder: " "${log_folder_current}"
    fi
    for data_split in "${data_splits[@]}"; do
        args_file=${args_folder}${method}"_args_15K.json"
        cur_time="$(date +%Y%m%d%H%M%S)"
        log_div=${data_split//\//_}
        CUDA_VISIBLE_DEVICES=${gpu} python3 ${py_code} "${args_file}" "${data_name}" "${data_split}" > "${log_folder_current}""${method}"_"${data_name}"_"${log_div}""${cur_time}"
    done
done
