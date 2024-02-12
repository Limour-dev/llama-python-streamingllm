def init(cfg):
    # ========== 自定义组合建的功能 ==========
    combine = cfg['btn_submit_vo_suggest_combine']
    tmp = cfg['btn_submit_vo_suggest'].click(**cfg['btn_start'])
    for f in combine:
        tmp = tmp.success(**cfg[f])
    tmp.success(**cfg['btn_finish'])
