from llama_cpp import *
from ctypes import POINTER, c_size_t
from llama_cpp._internals import (
    _LlamaModel,  # type: ignore
    _LlamaContext,  # type: ignore
    _LlamaBatch,  # type: ignore
    _LlamaTokenDataArray,  # type: ignore
)

from KMP_list import kmp_search, compute_lps_array
from Turbo_Colormap import map_value_to_color, NOCOLOR, LEGEND, BACK_WHITE


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

    def detokenize_sample_t(self):
        self.completion_tokens.append(self.token)
        all_text = self.str_detokenize(self.completion_tokens)
        if not all_text:
            return False
        self.t_bot.extend(self.completion_tokens)
        self.history += all_text
        self.completion_tokens = []
        return True

    def eval_sample_t(self):
        return self._eval_t([self.token])

    def endswith_t(self, token_list):
        return self.token in token_list

    def endswith_s(self, start_func, str_list, com_func=str.rstrip):
        if self.completion_tokens:  # 不完整
            return False

        history = self.history
        t_bot = self.t_bot

        if start_func(history):
            history = com_func(history)
            for x in str_list:
                if history.endswith(x):
                    n = len(t_bot)
                    for i in range(1, n):  # 找出需要弃置的tokens长度
                        tmp = self.str_detokenize(t_bot[n - i:])
                        tmp = com_func(tmp)
                        if tmp.endswith(x):
                            if i > 1:  # 最后一个token并未进入kv_cache
                                self.venv_pop_token(i - 1)
                            if history.endswith(tmp):
                                self.history = history[:-len(tmp)]  # 移除末尾的tmp
                            return True
        return False


kv_cache_type = {
    'f32': 0,
    'f16': 1,
    'q8_0': 8,
    'q4_0': 2,
    'q4_1': 3,
    'iq4_nl': 20,
    'q5_0': 6,
    'q5_1': 7
}

class StreamingLLM(Llama):

    __backend_initialized = False

    def __init__(
        self,
        model_path: str,
        *,
        # Model Params
        n_gpu_layers: int = 0,
        split_mode: int = llama_cpp.LLAMA_SPLIT_MODE_LAYER,
        main_gpu: int = 0,
        tensor_split: Optional[List[float]] = None,
        vocab_only: bool = False,
        use_mmap: bool = True,
        use_mlock: bool = False,
        kv_overrides: Optional[Dict[str, Union[bool, int, float]]] = None,
        # Context Params
        seed: int = llama_cpp.LLAMA_DEFAULT_SEED,
        n_ctx: int = 512,
        n_batch: int = 512,
        n_threads: Optional[int] = None,
        n_threads_batch: Optional[int] = None,
        rope_scaling_type: Optional[int] = llama_cpp.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        pooling_type: int = llama_cpp.LLAMA_POOLING_TYPE_UNSPECIFIED,
        rope_freq_base: float = 0.0,
        rope_freq_scale: float = 0.0,
        yarn_ext_factor: float = -1.0,
        yarn_attn_factor: float = 1.0,
        yarn_beta_fast: float = 32.0,
        yarn_beta_slow: float = 1.0,
        yarn_orig_ctx: int = 0,
        logits_all: bool = False,
        embedding: bool = False,
        offload_kqv: bool = True,
        # Sampling Params
        last_n_tokens_size: int = 64,
        # LoRA Params
        lora_base: Optional[str] = None,
        lora_scale: float = 1.0,
        lora_path: Optional[str] = None,
        # Backend Params
        numa: Union[bool, int] = False,
        # Chat Format Params
        chat_format: Optional[str] = None,
        chat_handler: Optional[llama_chat_format.LlamaChatCompletionHandler] = None,
        # Speculative Decoding
        draft_model: Optional[LlamaDraftModel] = None,
        # Tokenizer Override
        tokenizer: Optional[BaseLlamaTokenizer] = None,
        # Misc
        verbose: bool = True,
        # Extra Params
        type_k: str = 'f16',
        type_v: str = 'f16',
        **kwargs,  # type: ignore
    ):
        """Load a llama.cpp model from `model_path`.

        Examples:
            Basic usage

            >>> import llama_cpp
            >>> model = llama_cpp.Llama(
            ...     model_path="path/to/model",
            ... )
            >>> print(model("The quick brown fox jumps ", stop=["."])["choices"][0]["text"])
            the lazy dog

            Loading a chat model

            >>> import llama_cpp
            >>> model = llama_cpp.Llama(
            ...     model_path="path/to/model",
            ...     chat_format="llama-2",
            ... )
            >>> print(model.create_chat_completion(
            ...     messages=[{
            ...         "role": "user",
            ...         "content": "what is the meaning of life?"
            ...     }]
            ... ))

        Args:
            model_path: Path to the model.
            n_gpu_layers: Number of layers to offload to GPU (-ngl). If -1, all layers are offloaded.
            split_mode: How to split the model across GPUs. See llama_cpp.LLAMA_SPLIT_* for options.
            main_gpu: main_gpu interpretation depends on split_mode: LLAMA_SPLIT_NONE: the GPU that is used for the entire model. LLAMA_SPLIT_ROW: the GPU that is used for small tensors and intermediate results. LLAMA_SPLIT_LAYER: ignored
            tensor_split: How split tensors should be distributed across GPUs. If None, the model is not split.
            vocab_only: Only load the vocabulary no weights.
            use_mmap: Use mmap if possible.
            use_mlock: Force the system to keep the model in RAM.
            kv_overrides: Key-value overrides for the model.
            seed: RNG seed, -1 for random
            n_ctx: Text context, 0 = from model
            n_batch: Prompt processing maximum batch size
            n_threads: Number of threads to use for generation
            n_threads_batch: Number of threads to use for batch processing
            rope_scaling_type: RoPE scaling type, from `enum llama_rope_scaling_type`. ref: https://github.com/ggerganov/llama.cpp/pull/2054
            pooling_type: Pooling type, from `enum llama_pooling_type`.
            rope_freq_base: RoPE base frequency, 0 = from model
            rope_freq_scale: RoPE frequency scaling factor, 0 = from model
            yarn_ext_factor: YaRN extrapolation mix factor, negative = from model
            yarn_attn_factor: YaRN magnitude scaling factor
            yarn_beta_fast: YaRN low correction dim
            yarn_beta_slow: YaRN high correction dim
            yarn_orig_ctx: YaRN original context size
            logits_all: Return logits for all tokens, not just the last token. Must be True for completion to return logprobs.
            embedding: Embedding mode only.
            offload_kqv: Offload K, Q, V to GPU.
            last_n_tokens_size: Maximum number of tokens to keep in the last_n_tokens deque.
            lora_base: Optional path to base model, useful if using a quantized base model and you want to apply LoRA to an f16 model.
            lora_path: Path to a LoRA file to apply to the model.
            numa: numa policy
            chat_format: String specifying the chat format to use when calling create_chat_completion.
            chat_handler: Optional chat handler to use when calling create_chat_completion.
            draft_model: Optional draft model to use for speculative decoding.
            tokenizer: Optional tokenizer to override the default tokenizer from llama.cpp.
            verbose: Print verbose output to stderr.

        Raises:
            ValueError: If the model path does not exist.

        Returns:
            A Llama instance.
        """
        self.verbose = verbose

        set_verbose(verbose)

        if not StreamingLLM.__backend_initialized:
            with suppress_stdout_stderr(disable=verbose):
                llama_cpp.llama_backend_init()
            StreamingLLM.__backend_initialized = True

        if isinstance(numa, bool):
            self.numa = (
                llama_cpp.GGML_NUMA_STRATEGY_DISTRIBUTE
                if numa
                else llama_cpp.GGML_NUMA_STRATEGY_DISABLED
            )
        else:
            self.numa = numa

        if self.numa != llama_cpp.GGML_NUMA_STRATEGY_DISABLED:
            with suppress_stdout_stderr(disable=verbose):
                llama_cpp.llama_numa_init(self.numa)

        self.model_path = model_path

        # Model Params
        self.model_params = llama_cpp.llama_model_default_params()
        self.model_params.n_gpu_layers = (
            0x7FFFFFFF if n_gpu_layers == -1 else n_gpu_layers
        )  # 0x7FFFFFFF is INT32 max, will be auto set to all layers
        self.model_params.split_mode = split_mode
        self.model_params.main_gpu = main_gpu
        self.tensor_split = tensor_split
        self._c_tensor_split = None
        if self.tensor_split is not None:
            if len(self.tensor_split) > llama_cpp.LLAMA_MAX_DEVICES:
                raise ValueError(
                    f"Attempt to split tensors that exceed maximum supported devices. Current LLAMA_MAX_DEVICES={llama_cpp.LLAMA_MAX_DEVICES}"
                )
            # Type conversion and expand the list to the length of LLAMA_MAX_DEVICES
            FloatArray = ctypes.c_float * llama_cpp.LLAMA_MAX_DEVICES
            self._c_tensor_split = FloatArray(
                *tensor_split  # type: ignore
            )  # keep a reference to the array so it is not gc'd
            self.model_params.tensor_split = self._c_tensor_split
        self.model_params.vocab_only = vocab_only
        self.model_params.use_mmap = use_mmap if lora_path is None else False
        self.model_params.use_mlock = use_mlock

        # kv_overrides is the original python dict
        self.kv_overrides = kv_overrides
        if kv_overrides is not None:
            # _kv_overrides_array is a ctypes.Array of llama_model_kv_override Structs
            kvo_array_len = len(kv_overrides) + 1  # for sentinel element
            self._kv_overrides_array = (
                llama_cpp.llama_model_kv_override * kvo_array_len
            )()

            for i, (k, v) in enumerate(kv_overrides.items()):
                self._kv_overrides_array[i].key = k.encode("utf-8")
                if isinstance(v, bool):
                    self._kv_overrides_array[i].tag = llama_cpp.LLAMA_KV_OVERRIDE_TYPE_BOOL
                    self._kv_overrides_array[i].value.bool_value = v
                elif isinstance(v, int):
                    self._kv_overrides_array[i].tag = llama_cpp.LLAMA_KV_OVERRIDE_TYPE_INT
                    self._kv_overrides_array[i].value.int_value = v
                elif isinstance(v, float):
                    self._kv_overrides_array[i].tag = llama_cpp.LLAMA_KV_OVERRIDE_TYPE_FLOAT
                    self._kv_overrides_array[i].value.float_value = v
                else:
                    raise ValueError(f"Unknown value type for {k}: {v}")

            self._kv_overrides_array[-1].key = (
                b"\0"  # ensure sentinel element is zeroed
            )
            self.model_params.kv_overrides = self._kv_overrides_array

        self.n_batch = min(n_ctx, n_batch)  # ???
        self.n_threads = n_threads or max(multiprocessing.cpu_count() // 2, 1)
        self.n_threads_batch = n_threads_batch or max(
            multiprocessing.cpu_count() // 2, 1
        )

        # Context Params
        self.context_params = llama_cpp.llama_context_default_params()
        self.context_params.seed = seed
        self.context_params.n_ctx = n_ctx
        self.context_params.n_batch = self.n_batch
        self.context_params.n_threads = self.n_threads
        self.context_params.n_threads_batch = self.n_threads_batch
        self.context_params.rope_scaling_type = (
            rope_scaling_type
            if rope_scaling_type is not None
            else llama_cpp.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED
        )
        self.context_params.pooling_type = pooling_type
        self.context_params.rope_freq_base = (
            rope_freq_base if rope_freq_base != 0.0 else 0
        )
        self.context_params.rope_freq_scale = (
            rope_freq_scale if rope_freq_scale != 0.0 else 0
        )
        self.context_params.yarn_ext_factor = (
            yarn_ext_factor if yarn_ext_factor != 0.0 else 0
        )
        self.context_params.yarn_attn_factor = (
            yarn_attn_factor if yarn_attn_factor != 0.0 else 0
        )
        self.context_params.yarn_beta_fast = (
            yarn_beta_fast if yarn_beta_fast != 0.0 else 0
        )
        self.context_params.yarn_beta_slow = (
            yarn_beta_slow if yarn_beta_slow != 0.0 else 0
        )
        self.context_params.yarn_orig_ctx = yarn_orig_ctx if yarn_orig_ctx != 0 else 0
        self.context_params.logits_all = (
            logits_all if draft_model is None else True
        )  # Must be set to True for speculative decoding
        self.context_params.embeddings = embedding # TODO: Rename to embeddings

        #  KV cache quantization
        print(self.context_params.type_k, self.context_params.type_v)
        self.context_params.type_k = kv_cache_type[type_k]
        self.context_params.type_v = kv_cache_type[type_v]

        self.context_params.offload_kqv = offload_kqv

        # Sampling Params
        self.last_n_tokens_size = last_n_tokens_size

        self.cache: Optional[BaseLlamaCache] = None

        self.lora_base = lora_base
        self.lora_scale = lora_scale
        self.lora_path = lora_path

        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")

        self._model = _LlamaModel(
            path_model=self.model_path, params=self.model_params, verbose=self.verbose
        )

        # Override tokenizer
        self.tokenizer_ = tokenizer or LlamaTokenizer(self)

        # Set the default value for the context and correct the batch
        if n_ctx == 0:
            n_ctx = self._model.n_ctx_train()
            self.n_batch = min(n_ctx, n_batch)
            self.context_params.n_ctx = self._model.n_ctx_train()
            self.context_params.n_batch = self.n_batch

        self._ctx = _LlamaContext(
            model=self._model,
            params=self.context_params,
            verbose=self.verbose,
        )

        self._batch = _LlamaBatch(
            n_tokens=self.n_batch,
            embd=0,
            n_seq_max=self.context_params.n_ctx,
            verbose=self.verbose,
        )

        if self.lora_path:
            if self._model.apply_lora_from_file(
                self.lora_path,
                self.lora_scale,
                self.lora_base,
                self.n_threads,
            ):
                raise RuntimeError(
                    f"Failed to apply LoRA from lora path: {self.lora_path} to base path: {self.lora_base}"
                )

        if self.verbose:
            print(llama_cpp.llama_print_system_info().decode("utf-8"), file=sys.stderr)

        self.chat_format = chat_format
        self.chat_handler = chat_handler

        self.draft_model = draft_model

        self._n_vocab = self.n_vocab()
        self._n_ctx = self.n_ctx()

        self._token_nl = self.token_nl()
        self._token_eos = self.token_eos()

        self._candidates = _LlamaTokenDataArray(n_vocab=self._n_vocab)

        self.n_tokens = 0
        self.input_ids: npt.NDArray[np.intc] = np.ndarray((n_ctx,), dtype=np.intc)
        self.scores: npt.NDArray[np.single] = np.ndarray(
            (n_ctx, self._n_vocab), dtype=np.single
        )

        self._mirostat_mu = ctypes.c_float(
            2.0 * 5.0
        )  # TODO: Move this to sampling context

        try:
            self.metadata = self._model.metadata()
        except Exception as e:
            self.metadata = {}
            if self.verbose:
                print(f"Failed to load metadata: {e}", file=sys.stderr)

        if self.verbose:
            print(f"Model metadata: {self.metadata}", file=sys.stderr)

        if (
            self.chat_format is None
            and self.chat_handler is None
            and "tokenizer.chat_template" in self.metadata
        ):
            chat_format = llama_chat_format.guess_chat_format_from_gguf_metadata(
                self.metadata
            )

            if chat_format is not None:
                self.chat_format = chat_format
                if self.verbose:
                    print(f"Guessed chat format: {chat_format}", file=sys.stderr)
            else:
                template = self.metadata["tokenizer.chat_template"]
                try:
                    eos_token_id = int(self.metadata["tokenizer.ggml.eos_token_id"])
                except:
                    eos_token_id = self.token_eos()
                try:
                    bos_token_id = int(self.metadata["tokenizer.ggml.bos_token_id"])
                except:
                    bos_token_id = self.token_bos()

                eos_token = self._model.token_get_text(eos_token_id)
                bos_token = self._model.token_get_text(bos_token_id)

                if self.verbose:
                    print(f"Using gguf chat template: {template}", file=sys.stderr)
                    print(f"Using chat eos_token: {eos_token}", file=sys.stderr)
                    print(f"Using chat bos_token: {bos_token}", file=sys.stderr)

                self.chat_handler = llama_chat_format.Jinja2ChatFormatter(
                    template=template, eos_token=eos_token, bos_token=bos_token
                ).to_chat_handler()

        if self.chat_format is None and self.chat_handler is None:
            self.chat_format = "llama-2"
            if self.verbose:
                print(f"Using fallback chat format: {chat_format}", file=sys.stderr)
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

    def venv_viz(self):
        completion_tokens = []
        history = LEGEND + '\n'
        text_color = NOCOLOR
        for i in range(self.venv[-1]):
            idx = self.n_tokens - self.venv[-1] + i
            token = self._input_ids[idx]
            if not completion_tokens:  # 不完整则是第一个token
                # ========== 获取对应token的概率 ==========
                score = self.scores[idx-1: idx, :].ravel()  # 第i个token的分数是前i-1个token预测的，所以减一
                score = np.exp(score)  # 空白则全1，但无所谓了
                sum_score = np.sum(score)
                probabilities = score[token] / sum_score
                if probabilities < 0.001:
                    text_color = NOCOLOR
                else:
                    if text_color is NOCOLOR:
                        text_color = BACK_WHITE + map_value_to_color(probabilities)
                    else:
                        text_color = map_value_to_color(probabilities)
                history += text_color
            # ========== 避免不完整的utf-8编码 ==========
            completion_tokens.append(token)
            all_text = self.str_detokenize(completion_tokens)
            if not all_text:
                continue
            completion_tokens = []  # 完整则清空缓存
            history += repr(all_text)[1:-1]
        return history + NOCOLOR

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
        n_tokens = POINTER(c_size_t)(c_size_t(0))
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
