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
#       CREATED: 05/09/2016 23:30
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error

TRAIN="1m"
TEST="10k"

./clear.sh

parallel --gnu --results train/ --header : "./run.sh -m $TRAIN littmansoccer {left} {right} -t -L {left}-{right}" \
    ::: left minimaxq q phc wolf \
    ::: right random littmansoccerhandcoded minimaxq q phc wolf

parallel --gnu --results test/ --header : "./run.sh -m $TEST littmansoccer {left} {right}" \
    ::: left data/*-*.pickle \
    ::: right random littmansoccerhandcoded data/*-*.pickle

