def init(cfg):
    # ========== 共同 ==========
    model = cfg['model']
    btn_com = cfg['btn_com']
    s_info = cfg['s_info']
    lock = cfg['session_lock']
    # ========== 特殊 ==========
    chat_template = cfg['chat_template']
    chatbot = cfg['chatbot']
    chat_display_format = cfg['chat_display_format']

    # ========== 重新生成一份回答 ==========
    def btn_retry(history, _n_keep, _n_discard,
                  _temperature, _repeat_penalty, _frequency_penalty,
                  _presence_penalty, _repeat_last_n, _top_k,
                  _top_p, _min_p, _typical_p,
                  _tfs_z, _mirostat_mode, _mirostat_eta,
                  _mirostat_tau, _usr, _char,
                  _rag, _max_tokens):
        with lock:
            if not cfg['session_active']:
                raise RuntimeError
            # ========== 回滚到上一次用户输入 ==========
            if not model.venv_revision('usr'):
                yield history, model.venv_info
                return
            # ========== 模型输出 ==========
            model.venv_create('char')
            _tmp = btn_com(_n_keep, _n_discard,
                           _temperature, _repeat_penalty, _frequency_penalty,
                           _presence_penalty, _repeat_last_n, _top_k,
                           _top_p, _min_p, _typical_p,
                           _tfs_z, _mirostat_mode, _mirostat_eta,
                           _mirostat_tau, _char, _max_tokens)
            for _h in _tmp:
                history[-1][1] = _h
                yield history, model.venv_info
            # ========== 输出完毕后格式化输出 ==========
            history[-1][1] = chat_display_format(history[-1][1])
            yield history, model.venv_info

    cfg['btn_retry_fn'] = {
        'fn': btn_retry,
        'inputs': [chatbot]+cfg['setting'],
        'outputs': [chatbot, s_info],
    }
    cfg['btn_retry_fn'].update(cfg['btn_concurrency'])

    cfg['btn_retry'].click(
        **cfg['btn_start']
    ).success(
        **cfg['btn_retry_fn']
    ).success(
        **cfg['btn_finish']
    )