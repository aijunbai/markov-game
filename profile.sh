#!/bin/bash

set -o nounset                              # Treat unset variables as an error

source config.sh
PROF="main.prof"

profile() {
  PNG=`mktemp`
  $PYTHON gprof2dot.py -f pstats "$1" | dot -Tpng -o "$PNG"
  eog "$PNG"
  rm -f "$PNG"
  rm -f "$1"
}

time $PYTHON -m cProfile -o "$PROF" main.py $*
profile "$PROF"

