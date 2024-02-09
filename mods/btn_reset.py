def init(cfg):
    # ========== 共同 ==========
    model = cfg['model']
    s_info = cfg['s_info']

    def btn_debug(_cache_path):
        _tmp = model.load_session(_cache_path)
        print(f'load cache from {_cache_path} {_tmp}')
        return model.venv_info

    cfg['btn_reset'].click(
        fn=btn_debug,
        inputs=cfg['setting_cache_path'],
        outputs=s_info
    ).success(
        **cfg['btn_finish']
    )

    cfg['btn_debug'].click(
        fn=lambda: model.str_detokenize(model._input_ids),
        outputs=cfg['rag']
    )
