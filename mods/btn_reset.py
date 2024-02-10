def init(cfg):
    # ========== 共同 ==========
    model = cfg['model']
    s_info = cfg['s_info']

    def btn_reset(_cache_path):
        with cfg['session_lock']:
            _tmp = model.load_session(_cache_path)
            print(f'load cache from {_cache_path} {_tmp}')
            cfg['session_active'] = True
            return model.venv_info

    def btn_stop():
        cfg['btn_stop_status'] = True

    cfg['btn_stop'].click(
        fn=btn_stop
    )

    cfg['btn_reset'].click(
        fn=btn_stop
    ).success(
        fn=btn_reset,
        inputs=cfg['setting_cache_path'],
        outputs=s_info,
        **cfg['btn_concurrency']
    ).success(
        **cfg['btn_finish']
    )

    cfg['btn_debug'].click(
        fn=lambda: model.str_detokenize(model._input_ids),
        outputs=cfg['vo']
    )
