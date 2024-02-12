def init(cfg):
    # ========== 自定义组合建的功能 ==========
    combine = [
        'btn_submit_fn_usr',
        'btn_rag_fn',
        'btn_submit_fn_bot',
        'btn_vo_fn',
        'btn_suggest_fn'
    ]
    tmp = cfg['btn_submit_vo_suggest'].click(**cfg['btn_start'])
    for f in combine:
        tmp = tmp.success(**cfg[f])
    tmp.success(**cfg['btn_finish'])
