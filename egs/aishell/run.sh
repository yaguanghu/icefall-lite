#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export export PYTHONPATH=$PWD/../../:$PYTHONPATH

set -eou pipefail

nj=15
stage=0
stop_stage=10

. shared/parse_options.sh || exit 1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "stage 1: prepare data."
  ./local/prepare.sh --stage 0 --stop-stage 9
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "stage 2: training."
  export CUDA_VISIBLE_DEVICES=3
  python -m icefall_lite.bin.train --exp-dir zipformer_bbpe/exp
fi
