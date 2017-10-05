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

./run.sh -m $TRAIN littmansoccer minimaxq random -l -L MR &
./run.sh -m $TRAIN littmansoccer minimaxq littmansoccerhandcoded -l -L MH &
./run.sh -m $TRAIN littmansoccer minimaxq minimaxq -l -r -L MM &
./run.sh -m $TRAIN littmansoccer q random -l -L QR &
./run.sh -m $TRAIN littmansoccer q littmansoccerhandcoded -l -L QH &
./run.sh -m $TRAIN littmansoccer q q -l -r -L QQ &

wait

for i in MR MH MM QR QH QQ; do
    ./run.sh -m $TRAIN littmansoccer "data/${i}.pickle" q -r -R "${i}-challenger" &
done

wait

parallel --gnu --results test/ --header : "./run.sh -m $TEST littmansoccer {left} {right}" \
    ::: left data/M?.pickle data/Q?.pickle \
    ::: right random littmansoccerhandcoded data/*.pickle

