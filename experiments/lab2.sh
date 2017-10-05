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

TRAIN="1m"
PROBLEM="littmansoccer"
NUMPLOTS="-1"

if [ $# -ge 1 ]; then
    PROBLEM="$1"
fi

if [ $# -ge 2 ]; then
    TRAIN="$2"
fi

./clear.sh

if [ $PROBLEM = "littmansoccer" ]; then
    parallel --gnu --results train/ --header : "./run.sh -n $NUMPLOTS $PROBLEM {left} {right} -t -m $TRAIN -L {left}-{right} -R {right}-{left}" \
        ::: left minimaxq q phc wolf \
        ::: right random littmansoccerhandcoded minimaxq q phc wolf
else
    parallel --gnu --results train/ --header : "./run.sh -n $NUMPLOTS $PROBLEM {left} {right} -t -m $TRAIN -L {left}-{right} -R {right}-{left}" \
        ::: left minimaxq q phc wolf \
        ::: right random minimaxq q phc wolf
fi

