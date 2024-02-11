def init(cfg):
    chat_template = cfg['chat_template']
    model = cfg['model']
    gr = cfg['gr']
    lock = cfg['session_lock']

    with gr.Row():
        cfg['btn_vo'] = gr.Button("旁白")
        cfg['btn_rag'] = gr.Button("RAG")
        cfg['btn_retry'] = gr.Button("Retry")
        cfg['btn_stop'] = gr.Button("Stop")
        cfg['btn_reset'] = gr.Button("Reset")
        cfg['btn_debug'] = gr.Button("Debug")
        cfg['btn_submit_vo_suggest'] = gr.Button("Submit&旁白&建议", variant="primary")
        cfg['btn_submit'] = gr.Button("Submit")
        cfg['btn_suggest'] = gr.Button("建议")

    cfg['btn_stop_status'] = True

    # ========== 流式输出函数 ==========
    def btn_com(_n_keep, _n_discard,
                _temperature, _repeat_penalty, _frequency_penalty,
                _presence_penalty, _repeat_last_n, _top_k,
                _top_p, _min_p, _typical_p,
                _tfs_z, _mirostat_mode, _mirostat_eta,
                _mirostat_tau, _role, _max_tokens):
        # ========== 初始化输出模版 ==========
        t_bot = chat_template(_role)
        completion_tokens = []  # 有可能多个 tokens 才能构成一个 utf-8 编码的文字
        history = ''
        # ========== 流式输出 ==========
        for token in model.generate_t(
                tokens=t_bot,
                n_keep=_n_keep,
                n_discard=_n_discard,
                im_start=chat_template.im_start_token,
                top_k=_top_k,
                top_p=_top_p,
                min_p=_min_p,
                typical_p=_typical_p,
                temp=_temperature,
                repeat_penalty=_repeat_penalty,
                repeat_last_n=_repeat_last_n,
                frequency_penalty=_frequency_penalty,
                presence_penalty=_presence_penalty,
                tfs_z=_tfs_z,
                mirostat_mode=_mirostat_mode,
                mirostat_tau=_mirostat_tau,
                mirostat_eta=_mirostat_eta,
        ):
            # ========== eos or nlnl 说明eos了 ==========
            if token in chat_template.eos or token == chat_template.nlnl:
                t_bot.extend(completion_tokens)
                print('token in eos', token)
                break
            # ========== 避免不完整的utf-8编码 ==========
            completion_tokens.append(token)
            all_text = model.str_detokenize(completion_tokens)
            if not all_text:
                continue
            t_bot.extend(completion_tokens)
            # ========== 流式输出 ==========
            history += all_text
            yield history
            # ========== \n role \n 结构说明eos了 ==========
            tmp = chat_template.eos_in_role(history, t_bot)
            if tmp:
                tmp -= 1  # 最后一个token并未进入kv_cache
                if tmp:
                    model.venv_pop_token(tmp)
                break
            # ========== \n\n 结构说明eos了 ==========
            tmp = chat_template.eos_in_nlnl(history, t_bot)
            if tmp:
                tmp -= 1  # 最后一个token并未进入kv_cache
                if tmp:
                    model.venv_pop_token(tmp)
                break
            # ========== 过长 or 按下了stop按钮 ==========
            if len(t_bot) > _max_tokens or cfg['btn_stop_status']:
                break
            completion_tokens = []
        # ========== 查看末尾的换行符 ==========
        print('history', repr(history))
        # ========== 给 kv_cache 加上输出结束符 ==========
        model.eval_t(chat_template.im_end_nl, _n_keep, _n_discard)
        t_bot.extend(chat_template.im_end_nl)

    cfg['btn_com'] = btn_com

    btn_start_or_finish_outputs = [cfg['btn_submit'], cfg['btn_vo'],
                                   cfg['btn_suggest'], cfg['btn_retry'],
                                   cfg['btn_submit_vo_suggest']]

    def btn_start_or_finish(finish):
        tmp = gr.update(interactive=finish)
        tmp = (tmp,) * len(btn_start_or_finish_outputs)

        def _inner():
            with lock:
                if cfg['session_active'] != finish:
                    raise RuntimeError('任务中断！请稍等或Reset，如已Reset，请忽略。')
                cfg['session_active'] = not cfg['session_active']
                yield tmp
                if finish and cfg['btn_stop_status']:
                    raise RuntimeError('Stop或Reset被按下，任务已中断！如非您所为，可能他人正在使用中！')
                cfg['btn_stop_status'] = finish

        return _inner

    cfg['btn_concurrency'] = {
        'trigger_mode': 'once',
        'concurrency_id': 'btn_com',
        'concurrency_limit': 1
    }

    cfg['btn_start'] = {
        'fn': btn_start_or_finish(False),
        'outputs': btn_start_or_finish_outputs
    }
    cfg['btn_start'].update(cfg['btn_concurrency'])

    cfg['btn_finish'] = {
        'fn': btn_start_or_finish(True),
        'outputs': btn_start_or_finish_outputs
    }
    cfg['btn_finish'].update(cfg['btn_concurrency'])

    cfg['setting'] = [cfg[x] for x in ('setting_n_keep', 'setting_n_discard',
                                       'setting_temperature', 'setting_repeat_penalty', 'setting_frequency_penalty',
                                       'setting_presence_penalty', 'setting_repeat_last_n', 'setting_top_k',
                                       'setting_top_p', 'setting_min_p', 'setting_typical_p',
                                       'setting_tfs_z', 'setting_mirostat_mode', 'setting_mirostat_eta',
                                       'setting_mirostat_tau', 'role_usr', 'role_char',
                                       'rag', 'setting_max_tokens')]
