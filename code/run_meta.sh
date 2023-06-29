#!/bin/bash

#  ps -axu | grep meta | awk -F " " '{print $2}' | xargs kill

BASE_INDEX_KEY=`date '+%Y%m%d_%H%M%S'`
function run_a_method() {
  method=$2
  expt=$4
  dataset=$6
  gpu_id=$8

  INDEX_KEY="${BASE_INDEX_KEY}_${method}_${dataset}_${expt}"

  echo "$INDEX_KEY $@"
  nohup python META.py --index_key $INDEX_KEY $@ &
}

SEEDS="2"

#############################################################################################################
##################### MetaPrompter
lambdax=0.5
base_lr=0.1
meta_lr=0.001
method=MetaPrompter
eval_ft_step=5

####### 5w1s
run_a_method --method ${method} --expt "5way1shot" --ds "Reuters" --gpu_id 0 --lambdax $lambdax --base_lr $base_lr --eval_ft_step $eval_ft_step --meta_lr $meta_lr --seeds $SEEDS --job_type "PROD" &
run_a_method --method ${method} --expt "5way1shot" --ds "News20" --gpu_id 1 --lambdax $lambdax --base_lr $base_lr --eval_ft_step $eval_ft_step --meta_lr $meta_lr --seeds $SEEDS --job_type "PROD" &
run_a_method --method ${method} --expt "5way1shot" --ds "Amazon" --gpu_id 6 --lambdax $lambdax --base_lr $base_lr --eval_ft_step $eval_ft_step --meta_lr $meta_lr --seeds $SEEDS --job_type "PROD" &
run_a_method --method ${method} --expt "5way1shot" --ds "HuffPost" --gpu_id 3 --lambdax $lambdax --base_lr $base_lr --eval_ft_step 15 --meta_lr $meta_lr --seeds $SEEDS --job_type "PROD" &
run_a_method --method ${method} --expt "5way1shot" --ds "Hwu64" --gpu_id 4 --lambdax $lambdax --base_lr $base_lr --eval_ft_step $eval_ft_step --meta_lr $meta_lr --seeds $SEEDS --job_type "PROD" &
run_a_method --method ${method} --expt "5way1shot" --ds "Liu54" --gpu_id 5 --lambdax $lambdax --base_lr $base_lr --eval_ft_step $eval_ft_step --meta_lr $meta_lr --seeds $SEEDS --job_type "PROD" &

###### 5w5s
#run_a_method --method ${method} --expt "5way5shot" --ds "Reuters" --gpu_id 3 --lambdax $lambdax --base_lr $base_lr --eval_ft_step $eval_ft_step --meta_lr $meta_lr --seeds $SEEDS --job_type "PROD" &
#run_a_method --method ${method} --expt "5way5shot" --ds "News20" --gpu_id 4 --lambdax $lambdax --base_lr $base_lr --eval_ft_step $eval_ft_step --meta_lr $meta_lr --seeds $SEEDS --job_type "PROD" &
#run_a_method --method ${method} --expt "5way5shot" --ds "Amazon" --gpu_id 7 --lambdax $lambdax --base_lr $base_lr --eval_ft_step $eval_ft_step --meta_lr $meta_lr --seeds $SEEDS --job_type "PROD" &
#run_a_method --method ${method} --expt "5way5shot" --ds "HuffPost" --gpu_id 0 --lambdax $lambdax --base_lr $base_lr --eval_ft_step $eval_ft_step --meta_lr $meta_lr --seeds $SEEDS --job_type "PROD" &
#run_a_method --method ${method} --expt "5way5shot" --ds "Hwu64" --gpu_id 1 --lambdax $lambdax --base_lr $base_lr --eval_ft_step $eval_ft_step --meta_lr $meta_lr --seeds $SEEDS --job_type "PROD" &
#run_a_method --method ${method} --expt "5way5shot" --ds "Liu54" --gpu_id 2 --lambdax $lambdax --base_lr $base_lr --eval_ft_step $eval_ft_step --meta_lr $meta_lr --seeds $SEEDS --job_type "PROD" &
