#!/bin/bash - 
#===============================================================================
#
#          FILE: test.sh
# 
#         USAGE: ./test.sh 
# 
#   DESCRIPTION: 
# 
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: YOUR NAME (), 
#  ORGANIZATION: 
#       CREATED: 05/13/2016 16:04
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error

TRAIN="10m"
PROBLEM="littmansoccer"
NUMPLOTS="-1"

if [ $# -ge 1 ]; then
    PROBLEM="$1"
fi

if [ $# -ge 2 ]; then
    TRAIN="$2"
fi

./clear.sh

./run.sh -n $NUMPLOTS $PROBLEM minimaxq random -t -m $TRAIN -L MR &
./run.sh -n $NUMPLOTS $PROBLEM minimaxq q -t -m $TRAIN -L MQ &
./run.sh -n $NUMPLOTS $PROBLEM minimaxq minimaxq -t -m $TRAIN -L MM &
./run.sh -n $NUMPLOTS $PROBLEM q random -t -m $TRAIN -L QR &
./run.sh -n $NUMPLOTS $PROBLEM q q -t -m $TRAIN -L QQ &

if [ $PROBLEM = "littmansoccer" ]; then
    ./run.sh -n $NUMPLOTS $PROBLEM minimaxq littmansoccerhandcoded -t -m $TRAIN -L MH &
    ./run.sh -n $NUMPLOTS $PROBLEM q littmansoccerhandcoded -t -m $TRAIN -L QH &
fi

