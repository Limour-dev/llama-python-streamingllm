def init(cfg):
    cfg['btn_submit_vo_suggest'].click(
        **cfg['btn_start']
    ).success(
        **cfg['btn_submit_fn_usr']
    ).success(
        **cfg['btn_rag_fn']
    ).success(
        **cfg['btn_submit_fn_bot']
    ).success(
        **cfg['btn_vo_fn']
    ).success(
        **cfg['btn_suggest_fn']
    ).success(
        **cfg['btn_finish']
    )