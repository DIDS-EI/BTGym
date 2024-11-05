#!/bin/bash

/home/cxl/code/BTP/btp/planning/downward/fast-downward.py \
 --plan-file plan \
/home/cxl/code/BTP/examples/bddl/domain_omnigibson.bddl \
/home/cxl/code/BTP/activity_definitions/adding_chemicals_to_hot_tub/problem0.bddl \
 --search "astar(lmcut())"


