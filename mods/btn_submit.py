def init(cfg):
    # ========== 共同 ==========
    model = cfg['model']
    btn_com = cfg['btn_com']
    s_info = cfg['s_info']

    # ========== 特殊 ==========
    chat_template = cfg['chat_template']
    msg = cfg['msg']
    chatbot = cfg['chatbot']
    chat_display_format = cfg['chat_display_format']

    # ========== 显示用户消息 ==========
    def btn_submit_usr(message: str, history):
        # print('btn_submit_usr', message, history)
        if history is None:
            history = []
        return "", history + [[message.strip(), '']]

    # ========== 模型流式响应 ==========
    def btn_submit_bot(history, _n_keep, _n_discard,
                       _temperature, _repeat_penalty, _frequency_penalty,
                       _presence_penalty, _repeat_last_n, _top_k,
                       _top_p, _min_p, _typical_p,
                       _tfs_z, _mirostat_mode, _mirostat_eta,
                       _mirostat_tau, _usr, _char,
                       _rag, _max_tokens):
        # ========== 需要临时注入的内容 ==========
        if len(_rag) > 0:
            model.venv_create('rag')  # 记录 venv_idx
            t_rag = chat_template('system', _rag)
            model.eval_t(t_rag, _n_keep, _n_discard)
        # ========== 释放不再需要的环境 ==========
        model.venv_disband({'usr', 'char'})
        print('venv_disband char', model.venv_info)
        # ========== 用户输入 ==========
        model.venv_create('usr')
        t_msg = history[-1][0]
        t_msg = chat_template(_usr, t_msg)
        model.eval_t(t_msg, _n_keep, _n_discard)
        yield history, model.venv_info
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
        # ========== 响应完毕后清除注入的内容 ==========
        model.venv_remove('rag')  # 销毁对应的 venv
        yield history, model.venv_info

    cfg['btn_submit'].click(
        **cfg['btn_start']
    ).success(
        fn=btn_submit_usr, api_name="submit",
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    ).success(
        fn=btn_submit_bot,
        inputs=[chatbot]+cfg['setting'],
        outputs=[chatbot, s_info]
    ).success(
        **cfg['btn_finish']
    )
