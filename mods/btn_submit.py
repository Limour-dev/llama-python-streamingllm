def init(cfg):
    # ========== 共同 ==========
    model = cfg['model']
    btn_com = cfg['btn_com']
    s_info = cfg['s_info']
    lock = cfg['session_lock']

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
        with lock:
            if not cfg['session_active']:
                raise RuntimeError
            # ========== 释放不再需要的环境 ==========
            model.venv_disband({'usr', 'char'})
            print('venv_disband char', model.venv_info)
            # ========== 用户输入 ==========
            model.venv_create('usr')
            t_msg = history[-1][0]
            t_msg = chat_template(_usr, t_msg)
            model.eval_t(t_msg, _n_keep, _n_discard, chat_template.im_start_token)
            yield history, model.venv_info
            # ========== 模型输出 ==========
            if cfg['btn_stop_status']:
                return
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
            print(model.venv_viz())  # 调试用

    cfg['btn_submit_fn_usr'] = {
        'fn': btn_submit_usr,
        'inputs': [msg, chatbot],
        'outputs': [msg, chatbot]
    }
    cfg['btn_submit_fn_usr'].update(cfg['btn_concurrency'])

    cfg['btn_submit_fn_bot'] = {
        'fn': btn_submit_bot,
        'inputs': [chatbot]+cfg['setting'],
        'outputs': [chatbot, s_info],
    }
    cfg['btn_submit_fn_bot'].update(cfg['btn_concurrency'])

    cfg['btn_submit'].click(
        **cfg['btn_start']
    ).success(
        **cfg['btn_submit_fn_usr']
    ).success(
        **cfg['btn_submit_fn_bot']
    ).success(
        **cfg['btn_finish']
    )
