#!/bin/bash - 
#===============================================================================
#
#          FILE: clear.sh
# 
#         USAGE: ./clear.sh 
# 
#   DESCRIPTION: 
# 
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: YOUR NAME (), 
#  ORGANIZATION: 
#       CREATED: 05/03/2016 17:25
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error
rm -fr *.log *.prof nohup.out data/ result/

