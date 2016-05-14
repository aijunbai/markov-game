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

if [ $# -ge 1 ]; then
    PROBLEM="$1"
fi

./run.sh $PROBLEM minimaxq random -t -m $TRAIN -L MR &
./run.sh $PROBLEM minimaxq q -t -m $TRAIN -L MQ &
./run.sh $PROBLEM minimaxq minimaxq -t -m $TRAIN -L MM &
./run.sh $PROBLEM q random -t -m $TRAIN -L QR &
./run.sh $PROBLEM q q -t -m $TRAIN -L QQ &

