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

./run.sh -H $TRAIN littmansoccer minimaxq random -l -L MR &
./run.sh -H $TRAIN littmansoccer minimaxq littmansoccerhandcoded -l -L MH &
./run.sh -H $TRAIN littmansoccer minimaxq minimaxq -l -r -L MM &
./run.sh -H $TRAIN littmansoccer q random -l -L QR &
./run.sh -H $TRAIN littmansoccer q littmansoccerhandcoded -l -L QH &
./run.sh -H $TRAIN littmansoccer q q -l -r -L QQ &

wait

for i in MR MH MM QR QH QQ; do
    ./run.sh -H $TRAIN littmansoccer "data/${i}.pickle" q -r -R "${i}-challenger" &
done

wait

rm -f data/littmansoccer_*.pickle

parallel --gnu --results result/ --header : "./run.sh -H $TEST littmansoccer {left} {right}" \
    ::: left data/M?.pickle data/Q?.pickle \
    ::: right random littmansoccerhandcoded data/*.pickle

