def init(cfg):
    # ========== 共同 ==========
    model = cfg['model']
    btn_com = cfg['btn_com']
    s_info = cfg['s_info']
    lock = cfg['session_lock']

    # ========== 给用户提供默认回复的建议 ==========
    def btn_suggest(_n_keep, _n_discard,
                    _temperature, _repeat_penalty, _frequency_penalty,
                    _presence_penalty, _repeat_last_n, _top_k,
                    _top_p, _min_p, _typical_p,
                    _tfs_z, _mirostat_mode, _mirostat_eta,
                    _mirostat_tau, _usr, _char,
                    _rag, _max_tokens):
        with lock:
            if not cfg['session_active']:
                raise RuntimeError
            # ========== 模型输出建议 ==========
            model.venv_create('suggest')  # 创建隔离环境
            _tmp = btn_com(_n_keep, _n_discard,
                           _temperature, _repeat_penalty, _frequency_penalty,
                           _presence_penalty, _repeat_last_n, _top_k,
                           _top_p, _min_p, _typical_p,
                           _tfs_z, _mirostat_mode, _mirostat_eta,
                           _mirostat_tau, _usr, _max_tokens)
            _h = ''
            for _h in _tmp:
                yield _h, model.venv_info
            model.venv_remove('suggest')  # 销毁隔离环境
            yield _h, model.venv_info

    cfg['btn_suggest'].click(
        **cfg['btn_start']
    ).success(
        fn=btn_suggest,
        inputs=cfg['setting'],
        outputs=[cfg['msg'], s_info],
        **cfg['btn_concurrency']
    ).success(
        **cfg['btn_finish']
    )
