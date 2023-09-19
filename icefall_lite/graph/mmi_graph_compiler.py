import logging
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import k2
import torch

from icefall_lite.lexicon import UniqLexicon


class MmiTrainingGraphCompiler(object):

    def __init__(
        self,
        lang_dir: Path,
        uniq_filename: str = "uniq_lexicon.txt",
        device: Union[str, torch.device] = "cpu",
        oov: str = "<UNK>",
        sos_id: int = 1,
        eos_id: int = 1,
    ):
        """
        Args:
          lang_dir:
            Path to the lang directory. It is expected to contain the
            following files::

                - tokens.txt
                - words.txt
                - P.fst.txt

            The above files are generated by the script `prepare.sh`. You
            should have run it before running the training code.
          uniq_filename:
            File name to the lexicon in which every word has exactly one
            pronunciation. We assume this file is inside the given `lang_dir`.

          device:
            It indicates CPU or CUDA.
          oov:
            Out of vocabulary word. When a word in the transcript
            does not exist in the lexicon, it is replaced with `oov`.
        """
        self.lang_dir = Path(lang_dir)
        self.lexicon = UniqLexicon(lang_dir, uniq_filename=uniq_filename)
        self.device = torch.device(device)

        self.L_inv = self.lexicon.L_inv.to(self.device)

        self.oov_id = self.lexicon.word_table[oov]
        self.sos_id = sos_id
        self.eos_id = eos_id

        self.build_ctc_topo_P()

    def build_ctc_topo_P(self):
        """Built ctc_topo_P, the composition result of
        ctc_topo and P, where P is a pre-trained bigram
        word piece LM.
        """
        # Note: there is no need to save a pre-compiled P and ctc_topo
        # as it is very fast to generate them.
        logging.info(f"Loading P from {self.lang_dir/'P.fst.txt'}")
        with open(self.lang_dir / "P.fst.txt") as f:
            # P is not an acceptor because there is
            # a back-off state, whose incoming arcs
            # have label #0 and aux_label 0 (i.e., <eps>).
            P = k2.Fsa.from_openfst(f.read(), acceptor=False)

        first_token_disambig_id = self.lexicon.token_table["#0"]

        # P.aux_labels is not needed in later computations, so
        # remove it here.
        del P.aux_labels
        # CAUTION: The following line is crucial.
        # Arcs entering the back-off state have label equal to #0.
        # We have to change it to 0 here.
        labels = P.labels.clone()
        labels[labels >= first_token_disambig_id] = 0
        P.labels = labels

        P = k2.remove_epsilon(P)
        P = k2.arc_sort(P)
        P = P.to(self.device)
        # Add epsilon self-loops to P because we want the
        # following operation "k2.intersect" to run on GPU.
        P_with_self_loops = k2.add_epsilon_self_loops(P)

        max_token_id = max(self.lexicon.tokens)
        logging.info(f"Building ctc_topo (modified=False). max_token_id: {max_token_id}")
        ctc_topo = k2.ctc_topo(max_token_id, modified=False, device=self.device)

        ctc_topo_inv = k2.arc_sort(ctc_topo.invert_())

        logging.info("Building ctc_topo_P")
        ctc_topo_P = k2.intersect(ctc_topo_inv, P_with_self_loops, treat_epsilons_specially=False).invert()

        self.ctc_topo_P = k2.arc_sort(ctc_topo_P)
        logging.info(f"ctc_topo_P num_arcs: {self.ctc_topo_P.num_arcs}")

    def compile(self, texts: Iterable[str], replicate_den: bool = True) -> Tuple[k2.Fsa, k2.Fsa]:
        """Create numerator and denominator graphs from transcripts
        and the bigram phone LM.

        Args:
          texts:
            A list of transcripts. Within a transcript, words are
            separated by spaces. An example `texts` is given below::

                ["Hello icefall", "LF-MMI training with icefall using k2"]

          replicate_den:
            If True, the returned den_graph is replicated to match the number
            of FSAs in the returned num_graph; if False, the returned den_graph
            contains only a single FSA
        Returns:
          A tuple (num_graph, den_graph), where

            - `num_graph` is the numerator graph. It is an FsaVec with
              shape `(len(texts), None, None)`.

            - `den_graph` is the denominator graph. It is an FsaVec
              with the same shape of the `num_graph` if replicate_den is
              True; otherwise, it is an FsaVec containing only a single FSA.
        """
        transcript_fsa = self.build_transcript_fsa(texts)

        # remove word IDs from transcript_fsa since it is not needed
        del transcript_fsa.aux_labels
        # NOTE: You can comment out the above statement
        # if you want to run test/test_mmi_graph_compiler.py

        transcript_fsa_with_self_loops = k2.remove_epsilon_and_add_self_loops(transcript_fsa)

        transcript_fsa_with_self_loops = k2.arc_sort(transcript_fsa_with_self_loops)

        num = k2.compose(
            self.ctc_topo_P,
            transcript_fsa_with_self_loops,
            treat_epsilons_specially=False,
        )

        # CAUTION: Due to the presence of P,
        # the resulting `num` may not be connected
        num = k2.connect(num)

        num = k2.arc_sort(num)

        ctc_topo_P_vec = k2.create_fsa_vec([self.ctc_topo_P])
        if replicate_den:
            indexes = torch.zeros(len(texts), dtype=torch.int32, device=self.device)
            den = k2.index_fsa(ctc_topo_P_vec, indexes)
        else:
            den = ctc_topo_P_vec

        return num, den

    def build_transcript_fsa(self, texts: List[str]) -> k2.Fsa:
        """Convert transcripts to an FsaVec with the help of a lexicon
        and word symbol table.

        Args:
          texts:
            Each element is a transcript containing words separated by space(s).
            For instance, it may be 'HELLO icefall', which contains
            two words.

        Returns:
          Return an FST (FsaVec) corresponding to the transcript.
          Its `labels` is token IDs and `aux_labels` is word IDs.
        """
        word_ids_list = []
        for text in texts:
            word_ids = []
            for word in text.split():
                if word in self.lexicon.word_table:
                    word_ids.append(self.lexicon.word_table[word])
                else:
                    word_ids.append(self.oov_id)
            word_ids_list.append(word_ids)

        fsa = k2.linear_fsa(word_ids_list, self.device)
        fsa = k2.add_epsilon_self_loops(fsa)

        # The reason to use `invert_()` at the end is as follows:
        #
        # (1) The `labels` of L_inv is word IDs and `aux_labels` is token IDs
        # (2) `fsa.labels` is word IDs
        # (3) after intersection, the `labels` is still word IDs
        # (4) after `invert_()`, the `labels` is token IDs
        #     and `aux_labels` is word IDs
        transcript_fsa = k2.intersect(self.L_inv, fsa, treat_epsilons_specially=False).invert_()
        transcript_fsa = k2.arc_sort(transcript_fsa)
        return transcript_fsa

    def texts_to_ids(self, texts: List[str]) -> List[List[int]]:
        """Convert a list of texts to a list-of-list of piece IDs.

        Args:
          texts:
            It is a list of strings. Each string consists of space(s)
            separated words. An example containing two strings is given below:

                ['HELLO ICEFALL', 'HELLO k2']
            We assume it contains no OOVs. Otherwise, it will raise an
            exception.
        Returns:
          Return a list-of-list of token IDs.
        """
        return self.lexicon.texts_to_token_ids(texts).tolist()
