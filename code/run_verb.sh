#!/bin/bash

# ps -axu | grep verb | awk -F " " '{print $2}' | xargs kill

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
##################### RepVerbalizer
lambdax=1.0
lr=0.001
eval_ft_step=5
method=RepVerbalizer

####### 5way1shot
run_a_method --method $method --expt "5way1shot" --ds "Reuters" --gpu_id 0 --base_lr $lr --lambdax $lambdax --eval_ft_step $eval_ft_step --seeds $SEEDS --eval --job_type "PROD" &
run_a_method --method $method --expt "5way1shot" --ds "News20" --gpu_id 1 --base_lr $lr --lambdax $lambdax --eval_ft_step $eval_ft_step --seeds $SEEDS --eval --job_type "PROD" &
run_a_method --method $method --expt "5way1shot" --ds "Amazon" --gpu_id 2 --base_lr $lr --lambdax $lambdax --eval_ft_step $eval_ft_step --seeds $SEEDS --eval --job_type "PROD" &
run_a_method --method $method --expt "5way1shot" --ds "HuffPost" --gpu_id 3 --base_lr $lr --lambdax $lambdax --eval_ft_step $eval_ft_step --seeds $SEEDS --eval --job_type "PROD" &
run_a_method --method $method --expt "5way1shot" --ds "Hwu64" --gpu_id 4 --base_lr $lr --lambdax $lambdax --eval_ft_step $eval_ft_step --seeds $SEEDS --eval --job_type "PROD" &
run_a_method --method $method --expt "5way1shot" --ds "Liu54" --gpu_id 5 --base_lr $lr --lambdax $lambdax --eval_ft_step $eval_ft_step --seeds $SEEDS --eval --job_type "PROD" &

####### 5way5shot
#run_a_method --method $method --expt "5way5shot" --ds "Reuters" --gpu_id 3 --base_lr $lr --lambdax $lambdax --eval_ft_step $eval_ft_step --seeds $SEEDS --eval --job_type "PROD" &
#run_a_method --method $method --expt "5way5shot" --ds "News20" --gpu_id 4 --base_lr $lr --lambdax $lambdax --eval_ft_step $eval_ft_step --seeds $SEEDS --eval --job_type "PROD" &
#run_a_method --method $method --expt "5way5shot" --ds "Amazon" --gpu_id 5 --base_lr $lr --lambdax $lambdax --eval_ft_step $eval_ft_step --seeds $SEEDS --eval --job_type "PROD" &
#run_a_method --method $method --expt "5way5shot" --ds "HuffPost" --gpu_id 0 --base_lr $lr --lambdax $lambdax --eval_ft_step $eval_ft_step --seeds $SEEDS --eval --job_type "PROD" &
#run_a_method --method $method --expt "5way5shot" --ds "Hwu64" --gpu_id 1 --base_lr $lr --lambdax $lambdax --eval_ft_step $eval_ft_step --seeds $SEEDS --eval --job_type "PROD" &
#run_a_method --method $method --expt "5way5shot" --ds "Liu54" --gpu_id 2 --base_lr $lr --lambdax $lambdax --eval_ft_step $eval_ft_step --seeds $SEEDS --eval --job_type "PROD" &
