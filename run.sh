#!/bin/bash - 
#===============================================================================
#
#          FILE: run.sh
# 
#         USAGE: ./run.sh 
# 
#   DESCRIPTION: 
# 
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: YOUR NAME (), 
#  ORGANIZATION: 
#       CREATED: 03/23/2016 13:52
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error
PYTHON=`which python`
$PYTHON main.py
