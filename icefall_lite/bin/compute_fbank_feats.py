#!/usr/bin/env python3
# Copyright    2023  Gaotu Techedu Inc.  (authors: Yaguang Hu)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This file computes fbank features of the musan dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

import sentencepiece as spm
import torch
from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter, MonoCut, combine
from lhotse.recipes.utils import read_manifests_if_cached

from icefall_lite.utils import str2bool, filter_cuts

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="compute feats of dataset.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--srcdir", type=str, required=True, help="manifests data srcdir.")
    parser.add_argument("--outdir", type=str, required=True, help="fbank features output dir.")
    parser.add_argument("--num_mel_bins", type=int, default=80, help="Number of triangular mel-frequency bins.")
    parser.add_argument('--dataset_parts', nargs='+', help='datasets parts to compute', required=True)
    parser.add_argument("--prefix",
                        type=str,
                        help="Optional common prefix for the manifest files (underscore is automatically added).")
    parser.add_argument("--suffix", type=str, help="Optional common suffix for the manifest files ('json' by default)")
    parser.add_argument("--num_jobs", type=int, default=15, help="number of jobs for fbank computation.")
    parser.add_argument("--perturb-speed",
                        type=str2bool,
                        default=True,
                        help="Perturb speed with factor 0.9 and 1.1 on train subset.")
    parser.add_argument("--bpe-model", type=Path, help="Path to the bpe model")

    return parser


def is_cut_long(c: MonoCut) -> bool:
    return c.duration > 5


def compute_fbank_musan(
    srcdir: str,
    outdir: str,
    dataset_parts: List[str],
    prefix: str,
    suffix: str,
    num_mel_bins: int = 80,
    num_jobs: int = 15,
    bpe_model: Optional[str] = None,
    perturb_speed: Optional[bool] = True,
) -> None:

    src_dir = Path(srcdir)
    output_dir = Path(outdir)
    num_jobs = min(16, os.cpu_count())    # type: ignore

    prefix = "musan"
    suffix = "jsonl.gz"
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix=prefix,
        suffix=suffix,
    )
    assert manifests is not None

    assert len(manifests) == len(dataset_parts), (
        len(manifests),
        len(dataset_parts),
        list(manifests.keys()),
        dataset_parts,
    )

    musan_cuts_path = output_dir / "musan_cuts.jsonl.gz"

    if musan_cuts_path.is_file():
        logging.info(f"{musan_cuts_path} already exists - skipping")
        return

    logging.info("Extracting features for Musan")

    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    # create chunks of Musan with duration 5 - 10 seconds
    musan_cuts = (CutSet.from_manifests(recordings=combine(
        part["recordings"]
        for part in manifests.values())).cut_into_windows(10.0).filter(is_cut_long).compute_and_store_features(
            extractor=extractor,
            storage_path=f"{output_dir}/musan_feats",
            num_jobs=num_jobs,
            storage_type=LilcomChunkyWriter,
        ))
    musan_cuts.to_file(musan_cuts_path)


def compute_fbank(
    srcdir: str,
    outdir: str,
    dataset_parts: List[str],
    prefix: str,
    suffix: str,
    num_mel_bins: int = 80,
    num_jobs: int = 15,
    bpe_model: Optional[str] = None,
    perturb_speed: Optional[bool] = True,
) -> None:
    src_dir = Path(srcdir)
    output_dir = Path(outdir)
    num_jobs = min(num_jobs, os.cpu_count())    # type: ignore

    if bpe_model:
        logging.info(f"Loading {bpe_model}")
        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model)

    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix=prefix,
        suffix=suffix,
    )
    assert manifests is not None

    assert len(manifests) == len(dataset_parts), (
        len(manifests),
        len(dataset_parts),
        list(manifests.keys()),
        dataset_parts,
    )

    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    for partition, m in manifests.items():
        cuts_filename = f"{prefix}_cuts_{partition}.{suffix}"
        if (output_dir / cuts_filename).is_file():
            logging.info(f"{partition} already exists - skipping.")
            continue
        logging.info(f"Processing {partition}")
        cut_set = CutSet.from_manifests(
            recordings=m["recordings"],
            supervisions=m["supervisions"],
        )
        if "train" in partition:
            if bpe_model:
                cut_set = filter_cuts(cut_set, sp)
            if perturb_speed:
                logging.info(f"Doing speed perturb")
                cut_set = (cut_set + cut_set.perturb_speed(0.9) + cut_set.perturb_speed(1.1))
        cut_set = cut_set.compute_and_store_features(
            extractor=extractor,
            storage_path=f"{output_dir}/{prefix}_feats_{partition}",
            num_jobs=num_jobs,
            storage_type=LilcomChunkyWriter,
        )
        cut_set.to_file(output_dir / cuts_filename)


def main(cmd=None):
    parser = get_parser()
    args = parser.parse_args(cmd)
    assert Path(args.srcdir).is_dir()
    kwargs = vars(args)
    if args.prefix == "musan":
        compute_fbank_musan(**kwargs)
    else:
        compute_fbank(**kwargs)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
