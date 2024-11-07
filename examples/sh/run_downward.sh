#!/bin/bash

TASK_NAME=adding_chemicals_to_hot_tub

/home/cxl/code/BTGym/btgym/planning/downward/fast-downward.py \
--plan-file outputs/bddl_planning/$TASK_NAME \
--overall-time-limit 1 \
/home/cxl/code/BTGym/examples/bddl/domain_omnigibson.bddl \
/home/cxl/code/BTGym/activity_definitions/$TASK_NAME/problem0.bddl \
 --search "astar(lmcut())"


