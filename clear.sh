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
rm -fr *.log *.prof nohup.out

for d in data train test policy; do
    mv "${d}" "${d}_`date +%F_%H%M`"
    mkdir -p "${d}"
done

