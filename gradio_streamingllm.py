import hashlib
import os
import re
import json

import gradio as gr

from chat_template import ChatTemplate
from llama_cpp_python_streamingllm import StreamingLLM

#  ========== 让聊天界面的文本框等高 ==========
custom_css = r'''
#area > div {
    height: 100%;
}
#RAG-area {
    flex-grow: 1;
}
#RAG-area > label {
    height: 100%;
    display: flex;
    flex-direction: column;
}
#RAG-area > label > textarea {
    flex-grow: 1;
    max-height: 20vh;
}
#VO-area {
    flex-grow: 1;
}
#VO-area > label {
    height: 100%;
    display: flex;
    flex-direction: column;
}
#VO-area > label > textarea {
    flex-grow: 1;
    max-height: 20vh;
}
#prompt > label > textarea {
    max-height: 63px;
}
'''


#  ========== 适配 SillyTavern 的模版 ==========
def text_format(text: str, _env=None, **env):
    if _env is not None:
        for k, v in _env.items():
            text = text.replace(r'{{' + k + r'}}', v)
    for k, v in env.items():
        text = text.replace(r'{{' + k + r'}}', v)
    return text


# ========== 哈希函数 ==========
def x_hash(x: str):
    return hashlib.sha1(x.encode('utf-8')).hexdigest()


# ========== 读取配置文件 ==========
with open('rp_config.json', encoding='utf-8') as f:
    tmp = f.read()
with open('rp_sample_config.json', encoding='utf-8') as f:
    cfg = json.load(f)
cfg['setting_cache_path']['value'] += x_hash(tmp)
cfg.update(json.loads(tmp))

#  ========== 给引号加粗 ==========
reg_q = re.compile(r'“(.+?)”')


def chat_display_format(text: str):
    return reg_q.sub(r' **\g<0>** ', text)


#  ========== 温度、采样之类的设置 ==========
with gr.Blocks() as setting:
    with gr.Row():
        setting_path = gr.Textbox(label="模型路径", max_lines=1, scale=2, **cfg['setting_path'])
        setting_cache_path = gr.Textbox(label="缓存路径", max_lines=1, scale=2, **cfg['setting_cache_path'])
        setting_seed = gr.Number(label="随机种子", scale=1, **cfg['setting_seed'])
        setting_n_gpu_layers = gr.Number(label="n_gpu_layers", scale=1, **cfg['setting_n_gpu_layers'])
    with gr.Row():
        setting_ctx = gr.Number(label="上下文大小（Tokens）", **cfg['setting_ctx'])
        setting_max_tokens = gr.Number(label="最大响应长度（Tokens）", interactive=True, **cfg['setting_max_tokens'])
        setting_n_keep = gr.Number(value=10, label="n_keep", interactive=False)
        setting_n_discard = gr.Number(label="n_discard", interactive=True, **cfg['setting_n_discard'])
    with gr.Row():
        setting_temperature = gr.Number(label="温度", interactive=True, **cfg['setting_temperature'])
        setting_repeat_penalty = gr.Number(label="重复惩罚", interactive=True, **cfg['setting_repeat_penalty'])
        setting_frequency_penalty = gr.Number(label="频率惩罚", interactive=True, **cfg['setting_frequency_penalty'])
        setting_presence_penalty = gr.Number(label="存在惩罚", interactive=True, **cfg['setting_presence_penalty'])
        setting_repeat_last_n = gr.Number(label="惩罚范围", interactive=True, **cfg['setting_repeat_last_n'])
    with gr.Row():
        setting_top_k = gr.Number(label="Top-K", interactive=True, **cfg['setting_top_k'])
        setting_top_p = gr.Number(label="Top P", interactive=True, **cfg['setting_top_p'])
        setting_min_p = gr.Number(label="Min P", interactive=True, **cfg['setting_min_p'])
        setting_typical_p = gr.Number(label="Typical", interactive=True, **cfg['setting_typical_p'])
        setting_tfs_z = gr.Number(label="TFS", interactive=True, **cfg['setting_tfs_z'])
    with gr.Row():
        setting_mirostat_mode = gr.Number(label="Mirostat 模式", **cfg['setting_mirostat_mode'])
        setting_mirostat_eta = gr.Number(label="Mirostat 学习率", interactive=True, **cfg['setting_mirostat_eta'])
        setting_mirostat_tau = gr.Number(label="Mirostat 目标熵", interactive=True, **cfg['setting_mirostat_tau'])

    #  ========== 加载模型 ==========
    model = StreamingLLM(model_path=setting_path.value,
                         seed=setting_seed.value,
                         n_gpu_layers=setting_n_gpu_layers.value,
                         n_ctx=setting_ctx.value)
    setting_ctx.value = model.n_ctx()

# ========== 聊天的模版 默认 chatml ==========
eos = (2, 7)  # [eos, im_end]
chat_template = ChatTemplate(model)

# ========== 展示角色卡 ==========
with gr.Blocks() as role:
    with gr.Row():
        role_usr = gr.Textbox(label="用户名称", max_lines=1, interactive=False, **cfg['role_usr'])
        role_char = gr.Textbox(label="角色名称", max_lines=1, interactive=False, **cfg['role_char'])

    role_char_d = gr.Textbox(lines=10, label="故事描述", **cfg['role_char_d'])
    role_chat_style = gr.Textbox(lines=10, label="回复示例", **cfg['role_chat_style'])

    # model.eval_t([1])  # 这个暖机的 bos [1] 删了就不正常了
    if os.path.exists(setting_cache_path.value):
        # ========== 加载角色卡-缓存 ==========
        tmp = model.load_session(setting_cache_path.value)
        print(f'load cache from {setting_cache_path.value} {tmp}')
        tmp = chat_template('system',
                            text_format(role_char_d.value,
                                        char=role_char.value,
                                        user=role_usr.value))
        setting_n_keep.value = len(tmp)
        tmp = chat_template(role_char.value,
                            text_format(role_chat_style.value,
                                        char=role_char.value,
                                        user=role_usr.value))
        setting_n_keep.value += len(tmp)
        # ========== 加载角色卡-第一条消息 ==========
        chatbot = []
        for one in cfg["role_char_first"]:
            one['name'] = text_format(one['name'],
                                      char=role_char.value,
                                      user=role_usr.value)
            one['value'] = text_format(one['value'],
                                       char=role_char.value,
                                       user=role_usr.value)
            if one['name'] == role_char.value:
                chatbot.append((None, chat_display_format(one['value'])))
            print(one)
    else:
        # ========== 加载角色卡-角色描述 ==========
        tmp = chat_template('system',
                            text_format(role_char_d.value,
                                        char=role_char.value,
                                        user=role_usr.value))
        setting_n_keep.value = model.eval_t(tmp)  # 此内容永久存在

        # ========== 加载角色卡-回复示例 ==========
        tmp = chat_template(role_char.value,
                            text_format(role_chat_style.value,
                                        char=role_char.value,
                                        user=role_usr.value))
        setting_n_keep.value = model.eval_t(tmp)  # 此内容永久存在

        # ========== 加载角色卡-第一条消息 ==========
        chatbot = []
        for one in cfg["role_char_first"]:
            one['name'] = text_format(one['name'],
                                      char=role_char.value,
                                      user=role_usr.value)
            one['value'] = text_format(one['value'],
                                       char=role_char.value,
                                       user=role_usr.value)
            if one['name'] == role_char.value:
                chatbot.append((None, chat_display_format(one['value'])))
            print(one)
            tmp = chat_template(one['name'], one['value'])
            model.eval_t(tmp)  # 此内容随上下文增加将被丢弃

        # ========== 保存角色卡-缓存 ==========
        with open(setting_cache_path.value, 'wb') as f:
            pass
        tmp = model.save_session(setting_cache_path.value)
        print(f'save cache {tmp}')


# ========== 流式输出函数 ==========
def btn_submit_com(_n_keep, _n_discard,
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
        if token in eos:
            t_bot.extend(completion_tokens)
            break
        completion_tokens.append(token)
        all_text = model.str_detokenize(completion_tokens)
        if not all_text:
            continue
        t_bot.extend(completion_tokens)
        history += all_text
        yield history
        if len(t_bot) > _max_tokens or (all_text == '\n' and history[-2:-1] == '\n'):
            break
        completion_tokens = []
    # ========== 移除末尾的换行符 ==========
    if model.str_detokenize(t_bot[-2:]) == '\n\n':
        print('completion_tokens', completion_tokens)
        model.venv_pop_token()
    # ========== 给 kv_cache 加上输出结束符 ==========
    model.eval_t(chat_template.im_end_nl, _n_keep, _n_discard)
    t_bot.extend(chat_template.im_end_nl)


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
    rag_idx = None
    if len(_rag) > 0:
        rag_idx = model.venv_create()  # 记录 venv_idx
        t_rag = chat_template('system', _rag)
        model.eval_t(t_rag, _n_keep, _n_discard)
    model.venv_create()  # 与 t_rag 隔离
    # ========== 用户输入 ==========
    t_msg = history[-1][0]
    t_msg = chat_template(_usr, t_msg)
    model.eval_t(t_msg, _n_keep, _n_discard)
    # ========== 模型输出 ==========
    _tmp = btn_submit_com(_n_keep, _n_discard,
                          _temperature, _repeat_penalty, _frequency_penalty,
                          _presence_penalty, _repeat_last_n, _top_k,
                          _top_p, _min_p, _typical_p,
                          _tfs_z, _mirostat_mode, _mirostat_eta,
                          _mirostat_tau, _char, _max_tokens)
    for _h in _tmp:
        history[-1][1] = _h
        yield history, str((model.n_tokens, model.venv))
    # ========== 输出完毕后格式化输出 ==========
    history[-1][1] = chat_display_format(history[-1][1])
    yield history, str((model.n_tokens, model.venv))
    # ========== 及时清理上一次生成的旁白 ==========
    if vo_idx > 0:
        print('vo_idx', vo_idx, model.venv)
        model.venv_remove(vo_idx)
        print('vo_idx', vo_idx, model.venv)
        if rag_idx and vo_idx < rag_idx:
            rag_idx -= 1
    # ========== 响应完毕后清除注入的内容 ==========
    if rag_idx is not None:
        model.venv_remove(rag_idx)  # 销毁对应的 venv
    model.venv_disband()  # 退出隔离环境
    yield history, str((model.n_tokens, model.venv))
    print('venv_disband', vo_idx, model.venv)


# ========== 待实现 ==========
def btn_rag_(_rag, _msg):
    retn = ''
    return retn


vo_idx = 0


# ========== 输出一段旁白 ==========
def btn_submit_vo(_n_keep, _n_discard,
                  _temperature, _repeat_penalty, _frequency_penalty,
                  _presence_penalty, _repeat_last_n, _top_k,
                  _top_p, _min_p, _typical_p,
                  _tfs_z, _mirostat_mode, _mirostat_eta,
                  _mirostat_tau, _max_tokens):
    global vo_idx
    vo_idx = model.venv_create()  # 创建隔离环境
    # ========== 模型输出旁白 ==========
    _tmp = btn_submit_com(_n_keep, _n_discard,
                          _temperature, _repeat_penalty, _frequency_penalty,
                          _presence_penalty, _repeat_last_n, _top_k,
                          _top_p, _min_p, _typical_p,
                          _tfs_z, _mirostat_mode, _mirostat_eta,
                          _mirostat_tau, '旁白', _max_tokens)
    for _h in _tmp:
        yield _h, str((model.n_tokens, model.venv))


# ========== 给用户提供默认回复 ==========
def btn_submit_suggest(_n_keep, _n_discard,
                       _temperature, _repeat_penalty, _frequency_penalty,
                       _presence_penalty, _repeat_last_n, _top_k,
                       _top_p, _min_p, _typical_p,
                       _tfs_z, _mirostat_mode, _mirostat_eta,
                       _mirostat_tau, _usr, _max_tokens):
    model.venv_create()  # 创建隔离环境
    # ========== 模型输出 ==========
    _tmp = btn_submit_com(_n_keep, _n_discard,
                          _temperature, _repeat_penalty, _frequency_penalty,
                          _presence_penalty, _repeat_last_n, _top_k,
                          _top_p, _min_p, _typical_p,
                          _tfs_z, _mirostat_mode, _mirostat_eta,
                          _mirostat_tau, _usr, _max_tokens)
    _h = ''
    for _h in _tmp:
        yield _h, str((model.n_tokens, model.venv))
    model.venv_remove()  # 销毁隔离环境
    yield _h, str((model.n_tokens, model.venv))


# ========== 聊天页面 ==========
with gr.Blocks() as chatting:
    with gr.Row(equal_height=True):
        chatbot = gr.Chatbot(height='60vh', scale=2, value=chatbot,
                             avatar_images=(r'assets/user.png', r'assets/chatbot.webp'))
        with gr.Column(scale=1, elem_id="area"):
            rag = gr.Textbox(label='RAG', show_copy_button=True, elem_id="RAG-area")
            vo = gr.Textbox(label='VO', show_copy_button=True, elem_id="VO-area")
            s_info = gr.Textbox(value=str((model.n_tokens, model.venv)), max_lines=1, label='info', interactive=False)
    msg = gr.Textbox(label='Prompt', lines=2, max_lines=2, elem_id='prompt', autofocus=True, **cfg['msg'])
    with gr.Row():
        btn_rag = gr.Button("RAG")
        btn_submit = gr.Button("Submit")
        btn_retry = gr.Button("Retry")
        btn_com1 = gr.Button("自定义1")
        btn_com2 = gr.Button("自定义2")
        btn_com3 = gr.Button("自定义3")

    btn_rag.click(fn=btn_rag_, outputs=rag,
                  inputs=[rag, msg])

    btn_submit.click(
        fn=btn_submit_usr, api_name="submit",
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    ).then(
        fn=btn_submit_bot,
        inputs=[chatbot, setting_n_keep, setting_n_discard,
                setting_temperature, setting_repeat_penalty, setting_frequency_penalty,
                setting_presence_penalty, setting_repeat_last_n, setting_top_k,
                setting_top_p, setting_min_p, setting_typical_p,
                setting_tfs_z, setting_mirostat_mode, setting_mirostat_eta,
                setting_mirostat_tau, role_usr, role_char,
                rag, setting_max_tokens],
        outputs=[chatbot, s_info]
    ).then(
        fn=btn_submit_vo,
        inputs=[setting_n_keep, setting_n_discard,
                setting_temperature, setting_repeat_penalty, setting_frequency_penalty,
                setting_presence_penalty, setting_repeat_last_n, setting_top_k,
                setting_top_p, setting_min_p, setting_typical_p,
                setting_tfs_z, setting_mirostat_mode, setting_mirostat_eta,
                setting_mirostat_tau, setting_max_tokens],
        outputs=[vo, s_info]
    ).then(
        fn=btn_submit_suggest,
        inputs=[setting_n_keep, setting_n_discard,
                setting_temperature, setting_repeat_penalty, setting_frequency_penalty,
                setting_presence_penalty, setting_repeat_last_n, setting_top_k,
                setting_top_p, setting_min_p, setting_typical_p,
                setting_tfs_z, setting_mirostat_mode, setting_mirostat_eta,
                setting_mirostat_tau, role_usr, setting_max_tokens],
        outputs=[msg, s_info]
    )

    # ========== 用于调试 ==========
    btn_com1.click(fn=lambda: model.str_detokenize(model._input_ids), outputs=rag)


    @btn_com2.click(inputs=setting_cache_path,
                    outputs=s_info)
    def btn_com2(_cache_path):
        tmp = model.load_session(setting_cache_path.value)
        print(f'load cache from {setting_cache_path.value} {tmp}')
        global vo_idx
        vo_idx = 0
        model.venv = [0]
        return str((model.n_tokens, model.venv))

    # ========== 开始运行 ==========
demo = gr.TabbedInterface([chatting, setting, role],
                          ["聊天", "设置", '角色'],
                          css=custom_css)
gr.close_all()
demo.queue().launch(share=False)
