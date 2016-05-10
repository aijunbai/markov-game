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

H="1m"

./clear.sh

./run.sh -H $H littmansoccer minimaxq random -l -L MR &
./run.sh -H $H littmansoccer minimaxq minimaxq -l -r -L MM &
./run.sh -H $H littmansoccer q random -l -L QR &
./run.sh -H $H littmansoccer q q -l -r -L QQ &

wait

./run.sh -H $H littmansoccer data/MR.pickle q -r -R MR-challenger &
./run.sh -H $H littmansoccer data/MM.pickle q -r -R MM-challenger &
./run.sh -H $H littmansoccer data/QR.pickle q -r -R QR-challenger &
./run.sh -H $H littmansoccer data/QQ.pickle q -r -R QQ-challenger &

wait


rm -f data/littmansoccer_*.pickle

parallel --results result --header : "./run.sh -H 10k littmansoccer {left} {right}" \
    ::: left data/MR.pickle data/MM.pickle data/QR.pickle data/QQ.pickle \
    ::: right random littmansoccerhandcoded data/*.pickle

