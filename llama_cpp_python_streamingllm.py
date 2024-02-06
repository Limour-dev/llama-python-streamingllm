from typing import Optional, Sequence, Generator

from llama_cpp import Llama, LogitsProcessorList, LlamaGrammar, llama_cpp, npt, np, StoppingCriteriaList
from ctypes import POINTER

def is_UTF8_incomplete(all_text):
    multibyte_fix = 0
    if len(all_text) < 3:
        all_text = b'000' + all_text
    for k, char in enumerate(all_text[-3:]):
        k = 3 - k
        for num, pattern in [(2, 192), (3, 224), (4, 240)]:
            # Bitwise AND check
            if num > k and pattern & char == pattern:
                multibyte_fix = num - k
    return multibyte_fix


def get_complete_UTF8(all_text):
    multibyte_fix = is_UTF8_incomplete(all_text)
    if multibyte_fix > 0:
        multibyte_fix = multibyte_fix - 3
        return all_text[:multibyte_fix].decode("utf-8")
    else:
        return all_text.decode("utf-8")


class StreamingLLM(Llama):

    def str_detokenize(self, tokens) -> str:
        return get_complete_UTF8(self.detokenize(tokens))

    def kv_cache_seq_trim(self, start_pos):
        self._ctx.kv_cache_seq_rm(-1, start_pos, -1)
        self.n_tokens = start_pos

    def kv_cache_seq_ltrim(self, n_keep, n_discard=256, n_past=-1):
        self._ctx.kv_cache_seq_rm(-1, n_keep, n_keep + n_discard)
        if n_past < 0:
            n_past = self.n_tokens
        self._ctx.kv_cache_seq_shift(0, n_keep + n_discard, n_past, -n_discard)
        self.input_ids[n_keep:n_past - n_discard] = self.input_ids[n_keep + n_discard:n_past]
        self.n_tokens = n_past - n_discard

    def eval_t(self, tokens, n_keep=4, n_discard=256):
        if self._n_ctx < self.n_tokens + len(tokens):
            tmp_n_discard = max(n_discard, self.n_tokens + len(tokens) - self._n_ctx)
            self.kv_cache_seq_ltrim(n_keep, tmp_n_discard)
        # self._ctx.kv_cache_seq_rm(-1, self.n_tokens, -1)
        for i in range(0, len(tokens), self.n_batch):
            batch = tokens[i: i + self.n_batch]
            n_past = self.n_tokens
            n_tokens = len(batch)
            self._batch.set_batch(
                batch=batch, n_past=n_past, logits_all=self.context_params.logits_all
            )
            self._ctx.decode(self._batch)
            # Save tokens
            self.input_ids[n_past: n_past + n_tokens] = batch
            # Save logits
            rows = n_tokens
            cols = self._n_vocab
            offset = (
                0 if self.context_params.logits_all else n_tokens - 1
            )  # NOTE: Only save the last token logits if logits_all is False
            self.scores[n_past + offset: n_past + n_tokens, :].reshape(-1)[
            :
            ] = self._ctx.get_logits()[offset * cols: rows * cols]
            # Update n_tokens
            self.n_tokens += n_tokens
        return self.n_tokens

    def sample_t(
            self,
            top_k: int = 40,
            top_p: float = 0.95,
            min_p: float = 0.05,
            typical_p: float = 1.0,
            temp: float = 0.80,
            repeat_penalty: float = 1.1,
            repeat_last_n: int = 64,
            frequency_penalty: float = 0.0,
            presence_penalty: float = 0.0,
            tfs_z: float = 1.0,
            mirostat_mode: int = 0,
            mirostat_eta: float = 0.1,
            mirostat_tau: float = 5.0,
            penalize_nl: bool = True,
            logits_processor: Optional[LogitsProcessorList] = None,
            grammar: Optional[LlamaGrammar] = None,
    ):
        last_n_tokens_data = [llama_cpp.llama_token(0)] * max(
            0, repeat_last_n - self.n_tokens
        ) + self._input_ids[-repeat_last_n:].tolist()
        last_n_tokens_size = len(last_n_tokens_data)
        n_vocab = self._n_vocab
        n_ctx = self._n_ctx
        top_k = n_vocab if top_k <= 0 else top_k
        last_n_tokens_size = n_ctx if last_n_tokens_size < 0 else last_n_tokens_size
        last_n_tokens_data_c = (llama_cpp.llama_token * last_n_tokens_size)(
            *last_n_tokens_data
        )
        logits: npt.NDArray[np.single] = self.scores[self.n_tokens - 1: self.n_tokens, :].ravel()

        if logits_processor is not None:
            logits[:] = logits_processor(self._input_ids, logits)

        self._candidates.copy_logits(logits)
        self._ctx.sample_repetition_penalties(
            candidates=self._candidates,
            last_tokens_data=last_n_tokens_data_c,
            penalty_last_n=last_n_tokens_size,
            penalty_repeat=repeat_penalty,
            penalty_freq=frequency_penalty,
            penalty_present=presence_penalty,
        )
        if not penalize_nl:
            nl_logit = logits[self._token_nl]
            self._candidates.candidates.data[self._token_nl].logit = llama_cpp.c_float(
                nl_logit
            )

        if grammar is not None:
            self._ctx.sample_grammar(
                candidates=self._candidates,
                grammar=grammar,
            )

        if temp < 0.0:
            self._ctx.sample_softmax(candidates=self._candidates)
            id_ = self._candidates.candidates.data[0].id
        elif temp == 0.0:
            id_ = self._ctx.sample_token_greedy(candidates=self._candidates)
        elif mirostat_mode == 1:
            self._ctx.sample_temp(candidates=self._candidates, temp=temp)
            id_ = self._ctx.sample_token_mirostat(
                candidates=self._candidates,
                tau=mirostat_tau,
                eta=mirostat_eta,
                mu=2.0 * mirostat_tau,
                m=100,
            )
        elif mirostat_mode == 2:
            self._ctx.sample_temp(candidates=self._candidates, temp=temp)
            id_ = self._ctx.sample_token_mirostat_v2(
                candidates=self._candidates,
                tau=mirostat_tau,
                eta=mirostat_eta,
                mu=2.0 * mirostat_tau,
            )
        else:
            self._ctx.sample_top_k(candidates=self._candidates, k=top_k, min_keep=1)
            self._ctx.sample_tail_free(candidates=self._candidates, z=tfs_z, min_keep=1)
            self._ctx.sample_typical(
                candidates=self._candidates, p=typical_p, min_keep=1
            )
            self._ctx.sample_top_p(candidates=self._candidates, p=top_p, min_keep=1)
            self._ctx.sample_min_p(candidates=self._candidates, p=min_p, min_keep=1)
            self._ctx.sample_temp(candidates=self._candidates, temp=temp)
            id_ = self._ctx.sample_token(candidates=self._candidates)
        if grammar is not None:
            self._ctx.grammar_accept_token(grammar=grammar, token=id_)
        return id_

    def generate_t(
            self,
            tokens: Sequence[int],
            n_keep,
            n_discard: int = 256,
            top_k: int = 40,
            top_p: float = 0.95,
            min_p: float = 0.05,
            typical_p: float = 1.0,
            temp: float = 0.80,
            repeat_penalty: float = 1.1,
            repeat_last_n: int = 64,
            frequency_penalty: float = 0.0,
            presence_penalty: float = 0.0,
            tfs_z: float = 1.0,
            mirostat_mode: int = 0,
            mirostat_tau: float = 5.0,
            mirostat_eta: float = 0.1,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            grammar: Optional[LlamaGrammar] = None,
    ) -> Generator[int, Optional[Sequence[int]], None]:
        typical_p = float(typical_p)
        frequency_penalty = float(frequency_penalty)
        presence_penalty = float(presence_penalty)
        tfs_z = float(tfs_z)
        mirostat_tau = float(mirostat_tau)
        while True:
            self.eval_t(tokens, n_keep, n_discard)
            token = self.sample_t(
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                typical_p=typical_p,
                temp=temp,
                repeat_penalty=repeat_penalty,
                repeat_last_n=repeat_last_n,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                logits_processor=logits_processor,
                grammar=grammar,
            )
            if stopping_criteria is not None and stopping_criteria(
                    self._input_ids, self._scores[-1, :]
            ):
                return
            tokens_or_none = yield token
            tokens = [token]
            if tokens_or_none is not None:
                tokens.extend(tokens_or_none)

    def load_session(self, filepath: str):
        n_tokens = POINTER(llama_cpp.c_size_t)(llama_cpp.c_size_t(0))
        tokens = (llama_cpp.llama_token * self.n_ctx())()
        retn = llama_cpp.llama_load_session_file(self._ctx.ctx,
                                                 filepath.encode('utf-8'),
                                                 tokens,
                                                 self.n_ctx(),
                                                 n_tokens)
        self.n_tokens = n_tokens.contents.value
        self.input_ids[:self.n_tokens] = tokens[:self.n_tokens]
        return retn

    def save_session(self, filepath: str):
        # self.eval_t([0])  # https://github.com/ggerganov/llama.cpp/pull/4820
        # self.n_tokens -= 1
        tokens = self._input_ids.tolist()
        tokens = (llama_cpp.llama_token * len(tokens))(*tokens)
        return llama_cpp.llama_save_session_file(self._ctx.ctx, filepath.encode('utf-8'), tokens, self.n_tokens)
