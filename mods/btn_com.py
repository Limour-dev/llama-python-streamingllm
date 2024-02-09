def init(cfg):
    chat_template = cfg['chat_template']
    model = cfg['model']
    gr = cfg['gr']

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
            if token in chat_template.eos or token == chat_template.nlnl:
                t_bot.extend(completion_tokens)
                print('token in eos', token)
                break
            completion_tokens.append(token)
            all_text = model.str_detokenize(completion_tokens)
            if not all_text:
                continue
            t_bot.extend(completion_tokens)
            history += all_text
            yield history
            if token in chat_template.onenl:
                # ========== 移除末尾的换行符 ==========
                if t_bot[-2] in chat_template.onenl:
                    model.venv_pop_token()
                    break
                if t_bot[-2] in chat_template.onerl and t_bot[-3] in chat_template.onenl:
                    model.venv_pop_token()
                    break
            if history[-2:] == '\n\n':  # 各种 'x\n\n' 的token，比如'。\n\n'
                print('t_bot[-4:]', t_bot[-4:], repr(model.str_detokenize(t_bot[-4:])),
                      repr(model.str_detokenize(t_bot[-1:])))
                break
            if len(t_bot) > _max_tokens:
                break
            completion_tokens = []
        # ========== 查看末尾的换行符 ==========
        print('history', repr(history))
        # ========== 给 kv_cache 加上输出结束符 ==========
        model.eval_t(chat_template.im_end_nl, _n_keep, _n_discard)
        t_bot.extend(chat_template.im_end_nl)

    cfg['btn_com'] = btn_com

    def btn_start_or_finish(finish):
        tmp = gr.update(interactive=finish)

        def _inner():
            return tmp, tmp, tmp

        return _inner

    btn_start_or_finish_outputs = [cfg['btn_submit'], cfg['btn_vo'], cfg['btn_suggest']]

    cfg['btn_start'] = {
        'fn': btn_start_or_finish(False),
        'outputs': btn_start_or_finish_outputs
    }

    cfg['btn_finish'] = {
        'fn': btn_start_or_finish(True),
        'outputs': btn_start_or_finish_outputs
    }

    cfg['setting'] = [cfg[x] for x in ('setting_n_keep', 'setting_n_discard',
                                       'setting_temperature', 'setting_repeat_penalty', 'setting_frequency_penalty',
                                       'setting_presence_penalty', 'setting_repeat_last_n', 'setting_top_k',
                                       'setting_top_p', 'setting_min_p', 'setting_typical_p',
                                       'setting_tfs_z', 'setting_mirostat_mode', 'setting_mirostat_eta',
                                       'setting_mirostat_tau', 'role_usr', 'role_char',
                                       'rag', 'setting_max_tokens')]
