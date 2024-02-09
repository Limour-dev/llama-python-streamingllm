def init(cfg):
    # ========== 共同 ==========
    model = cfg['model']
    btn_com = cfg['btn_com']
    s_info = cfg['s_info']

    # ========== 输出一段旁白 ==========
    def btn_vo(_n_keep, _n_discard,
               _temperature, _repeat_penalty, _frequency_penalty,
               _presence_penalty, _repeat_last_n, _top_k,
               _top_p, _min_p, _typical_p,
               _tfs_z, _mirostat_mode, _mirostat_eta,
               _mirostat_tau, _usr, _char,
               _rag, _max_tokens):
        # ========== 及时清理上一次生成的旁白 ==========
        model.venv_remove('vo')
        print('清理旁白', model.venv_info)
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

    cfg['btn_vo'].click(
        **cfg['btn_start']
    ).success(
        fn=btn_vo,
        inputs=cfg['setting'],
        outputs=[cfg['vo'], s_info]
    ).success(
        **cfg['btn_finish']
    )
