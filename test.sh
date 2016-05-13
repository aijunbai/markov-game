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

./run.sh littmansoccer minimaxq random -t -m $TRAIN -L MR & 
./run.sh littmansoccer minimaxq q -t -m $TRAIN -L MQ & 
./run.sh littmansoccer minimaxq minimaxq -t -m $TRAIN -L MM & 
./run.sh littmansoccer q random -t -m $TRAIN -L QR & 
./run.sh littmansoccer q q -t -m $TRAIN -L QQ & 

