def init(cfg):
    # ========== 共同 ==========
    model = cfg['model']
    s_info = cfg['s_info']
    lock = cfg['session_lock']

    # ========== 特殊 ==========
    chat_template = cfg['chat_template']

    # ========== 注入内容 ==========
    def btn_rag(_n_keep, _n_discard,
                _temperature, _repeat_penalty, _frequency_penalty,
                _presence_penalty, _repeat_last_n, _top_k,
                _top_p, _min_p, _typical_p,
                _tfs_z, _mirostat_mode, _mirostat_eta,
                _mirostat_tau, _usr, _char,
                _rag, _max_tokens):
        with lock:
            if not cfg['session_active']:
                raise RuntimeError
            if cfg['btn_stop_status']:
                yield model.venv_info
                return
            # ========== 清除之前注入的内容 ==========
            model.venv_remove('rag')
            # ========== 没有需要注入的内容 ==========
            if not _rag:
                yield model.venv_info
                return
            # ========== 需要临时注入的内容 ==========
            model.venv_create('rag')
            t_rag = chat_template('system', _rag)
            model.eval_t(t_rag, _n_keep, _n_discard, chat_template.im_start_token)
            yield model.venv_info

    cfg['btn_rag_fn'] = {
        'fn': btn_rag,
        'inputs': cfg['setting'],
        'outputs': s_info
    }
    cfg['btn_rag_fn'].update(cfg['btn_concurrency'])

    cfg['btn_rag'].click(
        **cfg['btn_start']
    ).success(
        **cfg['btn_rag_fn']
    ).success(
        **cfg['btn_finish']
    )
