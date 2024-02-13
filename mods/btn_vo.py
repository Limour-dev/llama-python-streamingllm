def init(cfg):
    # ========== 共同 ==========
    model = cfg['model']
    btn_com = cfg['btn_com']
    s_info = cfg['s_info']
    lock = cfg['session_lock']

    # ========== 输出一段旁白 ==========
    def btn_vo(_n_keep, _n_discard,
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
                yield '', model.venv_info
                return
            # ========== 模型输出旁白 ==========
            model.venv_create('vo')  # 创建隔离环境
            _tmp = btn_com(_n_keep, _n_discard,
                           _temperature, _repeat_penalty, _frequency_penalty,
                           _presence_penalty, _repeat_last_n, _top_k,
                           _top_p, _min_p, _typical_p,
                           _tfs_z, _mirostat_mode, _mirostat_eta,
                           _mirostat_tau, '旁白', _max_tokens)
            for _h in _tmp:
                yield _h, model.venv_info
            # ========== 及时清理上一次生成的旁白 ==========
            model.venv_remove('vo', keep_last=1)
            yield _h, model.venv_info
            print('清理旁白', model.venv_info)

    cfg['btn_vo_fn'] = {
        'fn': btn_vo,
        'inputs': cfg['setting'],
        'outputs': [cfg['vo'], s_info]
    }
    cfg['btn_vo_fn'].update(cfg['btn_concurrency'])

    cfg['btn_vo'].click(
        **cfg['btn_start']
    ).success(
        **cfg['btn_vo_fn']
    ).success(
        **cfg['btn_finish']
    )
