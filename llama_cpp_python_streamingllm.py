from typing import Optional, Sequence, Generator

from llama_cpp import Llama, LogitsProcessorList, LlamaGrammar, llama_cpp, npt, np, StoppingCriteriaList
from ctypes import POINTER

from KMP_list import kmp_search, compute_lps_array


class LLMGenerate:
    def __init__(
            self,
            model,
            n_keep,
            n_discard: int = 256,
            im_start=None,
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
            mirostat_eta: float = 0.1
    ):
        def _eval_t(tokens):
            return model.eval_t(
                tokens=tokens,
                n_keep=n_keep,
                n_discard=n_discard,
                im_start=im_start
            )

        def _sample_t(logits_processor):
            return model.sample_t(
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
                logits_processor=logits_processor
            )

        self._eval_t = _eval_t
        self._sample_t = _sample_t
        self.str_detokenize = model.str_detokenize
        self.venv_pop_token = model.venv_pop_token
        # ========== 保存输出 ==========
        self.t_bot = []
        self.completion_tokens = []
        self.history = ''
        self.token = None

    def eval_t(self, tokens):
        # ========== 避免不完整的utf-8编码 ==========
        self.completion_tokens.extend(tokens)
        all_text = self.str_detokenize(self.completion_tokens)
        if all_text:
            self.t_bot.extend(self.completion_tokens)
            self.history += all_text
            self.completion_tokens = []
        return self._eval_t(tokens)

    def sample_t(self, logits_processor):
        self.token = self._sample_t(logits_processor)
        return self.token

    def endswith_t(self, token_list):
        return self.token in token_list

    def endswith_func(self, start_func, func_list, com_func=str.rstrip, eval_t=True):
        self.completion_tokens.append(self.token)
        all_text = self.str_detokenize(self.completion_tokens)
        if all_text:  # 完整了

            if eval_t:
                self.t_bot.extend(self.completion_tokens)  # 节省开销
                self.history += all_text
                self.completion_tokens = []

                history = self.history
                t_bot = self.t_bot
            else:
                history = self.history + all_text
                t_bot = self.t_bot + self.completion_tokens

            if start_func(history):
                history = com_func(history)
                for func in func_list:
                    if func(history):
                        n = len(t_bot)
                        for i in range(1, n):  # 找出需要弃置的tokens长度
                            tmp = self.str_detokenize(t_bot[n - i:])
                            tmp = com_func(tmp)
                            if func(tmp):
                                if i > 1:  # 最后一个token并未进入kv_cache
                                    self.venv_pop_token(i - 1)
                                if history.endswith(tmp):
                                    self.history = history[:-len(tmp)]  # 移除末尾的tmp
                                return True

        if eval_t:
            self._eval_t([self.token])  # 避免再执行一次 eval_t
        else:
            self.completion_tokens.pop()  # 删除 append 的 token
        return False


class StreamingLLM(Llama):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, **kwargs)
        self._venv_init()

    def str_detokenize(self, tokens) -> str:
        return self.detokenize(tokens).decode('utf-8', errors='ignore')

    def kv_cache_seq_trim(self):
        self._ctx.kv_cache_seq_rm(-1, self.n_tokens, -1)

    def _venv_init(self):
        self.venv = [0]
        self.venv_idx_map = []

    def venv_create(self, name: str):
        self.venv.append(0)
        self.venv_idx_map.append(name)
        return name

    def venv_disband(self, name_set):
        if len(self.venv) <= 1:
            return False
        name_set = {x for x in name_set if x in self.venv_idx_map}
        if not name_set:
            return False
        while self.venv_idx_map:
            if self.venv_idx_map[0] in name_set:
                self.venv_idx_map.pop(0)  # 删除
                tmp = self.venv.pop(1)  # 对应的 venv 移入上一层
                self.venv[0] += tmp
            else:
                break
        return True

    def venv_revision(self, name: str):
        if len(self.venv) <= 1:
            return False
        if name not in self.venv_idx_map:
            return False
        _s = 0
        while self.venv_idx_map:
            if self.venv_idx_map[-1] == name:
                break
            self.venv_idx_map.pop()  # 删除
            _s += self.venv.pop()
        if _s:
            self.n_tokens -= min(_s, self.n_tokens)
            self.kv_cache_seq_trim()
        return True

    def venv_remove(self, name: str, keep_last=0):
        if len(self.venv) <= 1:
            return False
        if name not in self.venv_idx_map:
            return False
        venv_idx = self.venv_idx_map.index(name) + 1
        count_name = self.venv_idx_map.count(name) if keep_last else 0
        while self.venv_idx_map:
            if keep_last and count_name <= keep_last:
                break  # 保留最后n个
            self.venv_idx_map.pop(venv_idx - 1)  # 删除
            if venv_idx == len(self.venv) - 1:
                # 最后一层
                self.n_tokens -= min(self.venv.pop(), self.n_tokens)
                self.kv_cache_seq_trim()
                break
            else:
                # 非最后一层
                n_keep = self.n_tokens - sum(self.venv[i] for i in range(venv_idx, len(self.venv)))
                n_discard = self.venv.pop(venv_idx)
                self.kv_cache_seq_ltrim(n_keep, n_discard)
                try:
                    venv_idx = self.venv_idx_map.index(name, venv_idx - 1) + 1
                except ValueError:  # 没有了
                    break
            count_name -= 1  # 计数减一
        return True

    def venv_pop_token(self, n=1):
        self.n_tokens -= n
        self.venv[-1] -= n
        self.kv_cache_seq_trim()

    @property
    def venv_info(self):
        return str((self.n_tokens, self.venv, self.venv_idx_map))

    def kv_cache_seq_ltrim(self, n_keep, n_discard=256, n_past=-1, im_start=None):
        if n_keep < 0:
            return
        if n_past < 0:
            n_past = self.n_tokens
        if im_start is not None:  # [<|im_start|>, name, nl]
            lps = compute_lps_array(im_start)
            _idx = kmp_search(self.input_ids, im_start, n_keep + n_discard, n_past, lps)
            if _idx >= n_keep:  # 其实是大于等于 n_keep + n_discard
                n_discard = _idx - n_keep  # 截断到最近的 im_start 序列结构
            else:
                _idx = kmp_search(self.input_ids, im_start, n_keep, n_past, lps)
                if _idx >= n_keep:
                    n_keep = _idx + len(im_start)  # 至少保留一个 im_start 序列结构
        self._ctx.kv_cache_seq_rm(-1, n_keep, n_keep + n_discard)
        self._ctx.kv_cache_seq_shift(0, n_keep + n_discard, n_past, -n_discard)
        self.input_ids[n_keep:n_past - n_discard] = self.input_ids[n_keep + n_discard:n_past]
        self.n_tokens = n_past - n_discard

    def eval_t(self, tokens, n_keep=4, n_discard=256, im_start=None):
        if self._n_ctx < self.n_tokens + len(tokens):
            tmp_n_discard = max(n_discard, self.n_tokens + len(tokens) - self._n_ctx)
            self.kv_cache_seq_ltrim(n_keep, tmp_n_discard, im_start=im_start)
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
            self.venv[-1] += n_tokens
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
            logits_processor=None,
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
            im_start=None,
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
            self.eval_t(tokens, n_keep, n_discard, im_start=im_start)
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
            tokens = yield token
            if tokens is None:
                tokens = [token]

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
        self._venv_init()
        return retn

    def save_session(self, filepath: str):
        tokens = self._input_ids.tolist()
        tokens = (llama_cpp.llama_token * len(tokens))(*tokens)
        return llama_cpp.llama_save_session_file(self._ctx.ctx, filepath.encode('utf-8'), tokens, self.n_tokens)
