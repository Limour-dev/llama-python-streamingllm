def init(cfg):
    # ========== 待实现 ==========
    def btn_rag_(_rag, _msg):
        retn = ''
        return retn

    cfg['btn_rag'].click(fn=btn_rag_, outputs=cfg['rag'],
                         inputs=[cfg['rag'], cfg['msg']])
