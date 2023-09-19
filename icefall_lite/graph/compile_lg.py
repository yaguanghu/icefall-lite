#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang, Wei Kang)
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
This script takes as input lang_dir and generates LG from

    - L, the lexicon, built from lang_dir/L_disambig.pt

        Caution: We use a lexicon that contains disambiguation symbols

    - G, the LM, built from data/lm/G_3_gram.fst.txt

The generated LG is saved in $lang_dir/LG.pt
"""
import argparse
import logging
from pathlib import Path

import k2
import torch

from icefall_lite.lexicon import Lexicon


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang-dir",
        type=str,
        help="""Input and output directory.
        """,
    )
    parser.add_argument(
        "--lm",
        type=str,
        default="G_3_gram",
        help="""Stem name for LM used in HLG compiling.
        """,
    )

    return parser.parse_args()


def compile_LG(lang_dir: str, lm: str = "G_3_gram") -> k2.Fsa:
    """
    Args:
      lang_dir:
        The language directory, e.g., data/lang_phone or data/lang_bpe_5000.

    Return:
      An FSA representing LG.
    """
    lexicon = Lexicon(lang_dir)
    L = k2.Fsa.from_dict(torch.load(f"{lang_dir}/L_disambig.pt"))

    if Path(f"data/lm/{lm}.pt").is_file():
        logging.info(f"Loading pre-compiled {lm}")
        d = torch.load(f"data/lm/{lm}.pt")
        G = k2.Fsa.from_dict(d)
    else:
        logging.info(f"Loading {lm}.fst.txt")
        with open(f"data/lm/{lm}.fst.txt") as f:
            G = k2.Fsa.from_openfst(f.read(), acceptor=False)
            torch.save(G.as_dict(), f"data/lm/{lm}.pt")

    first_token_disambig_id = lexicon.token_table["#0"]
    first_word_disambig_id = lexicon.word_table["#0"]

    L = k2.arc_sort(L)
    G = k2.arc_sort(G)

    logging.info("Intersecting L and G")
    LG = k2.compose(L, G)
    logging.info(f"LG shape: {LG.shape}")

    logging.info("Connecting LG")
    LG = k2.connect(LG)
    logging.info(f"LG shape after k2.connect: {LG.shape}")

    logging.info(type(LG.aux_labels))
    logging.info("Determinizing LG")

    LG = k2.determinize(LG, k2.DeterminizeWeightPushingType.kLogWeightPushing)
    logging.info(type(LG.aux_labels))

    logging.info("Connecting LG after k2.determinize")
    LG = k2.connect(LG)

    logging.info("Removing disambiguation symbols on LG")

    # LG.labels[LG.labels >= first_token_disambig_id] = 0
    # see https://github.com/k2-fsa/k2/pull/1140
    labels = LG.labels
    labels[labels >= first_token_disambig_id] = 0
    LG.labels = labels

    assert isinstance(LG.aux_labels, k2.RaggedTensor)
    LG.aux_labels.values[LG.aux_labels.values >= first_word_disambig_id] = 0

    LG = k2.remove_epsilon(LG)
    logging.info(f"LG shape after k2.remove_epsilon: {LG.shape}")

    LG = k2.connect(LG)
    LG.aux_labels = LG.aux_labels.remove_values_eq(0)

    logging.info("Arc sorting LG")
    LG = k2.arc_sort(LG)

    return LG


def main():
    args = get_args()
    lang_dir = Path(args.lang_dir)

    if (lang_dir / "LG.pt").is_file():
        logging.info(f"{lang_dir}/LG.pt already exists - skipping")
        return

    logging.info(f"Processing {lang_dir}")

    LG = compile_LG(lang_dir, args.lm)
    logging.info(f"Saving LG.pt to {lang_dir}")
    torch.save(LG.as_dict(), f"{lang_dir}/LG.pt")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
