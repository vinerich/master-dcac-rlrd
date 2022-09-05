#!/bin/bash

DELAY="${1:-1}"
MIN_DELAY="$(($DELAY-1))"

echo "Executing with DELAY $DELAY and MIN_DELAY $MIN_DELAY"

python -m rlrd run rlrd:DcacTraining Env.min_action_delay=$MIN_DELAY Env.sup_action_delay=$DELAY
