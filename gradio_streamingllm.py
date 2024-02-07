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
        setting_path = gr.Textbox(label="模型路径", scale=2, **cfg['setting_path'])
        setting_cache_path = gr.Textbox(label="缓存路径", scale=1, **cfg['setting_cache_path'])
        setting_seed = gr.Number(label="随机种子", scale=1, **cfg['setting_seed'])
        setting_n_gpu_layers = gr.Number(label="n_gpu_layers", scale=1, **cfg['setting_n_gpu_layers'])
    with gr.Row():
        setting_ctx = gr.Number(label="上下文大小（Tokens）", **cfg['setting_ctx'])
        setting_max_tokens = gr.Number(label="最大响应长度（Tokens）", interactive=True, **cfg['setting_max_tokens'])
    with gr.Row():
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
    setting_ctx.value = model.context_params.n_ctx

# ========== 聊天的模版 默认 chatml ==========
eos = (2, 7)  # [eos, im_end]
chat_template = ChatTemplate(model)

# ========== 展示角色卡 ==========
with gr.Blocks() as role:
    with gr.Row():
        role_usr = gr.Textbox(label="用户名称", interactive=False, **cfg['role_usr'])
        role_char = gr.Textbox(label="角色名称", interactive=False, **cfg['role_char'])

    role_char_d = gr.Textbox(lines=10, max_lines=99, label="角色描述", **cfg['role_char_d'])
    role_chat_style = gr.Textbox(lines=10, max_lines=99, label="回复示例", **cfg['role_chat_style'])
    role_char_first = gr.Textbox(lines=10, max_lines=99, label="第一条消息", **cfg['role_char_first'])

    # model.eval_t([1])  # 这个暖机的 bos [1] 删了就不正常了
    if os.path.exists(setting_cache_path.value):
        # ========== 加载角色卡-缓存 ==========
        tmp = model.load_session(setting_cache_path.value)
        print(f'load cache from {setting_cache_path.value} {tmp}')
        tmp = chat_template('system',
                            text_format(role_char_d.value,
                                        char=role_char.value,
                                        user=role_usr.value))
        setting_n_keep.value = 1 + len(tmp)
        tmp = chat_template(role_char.value,
                            text_format(role_chat_style.value,
                                        char=role_char.value,
                                        user=role_usr.value))
        setting_n_keep.value += len(tmp)
        tmp = text_format(role_char_first.value,
                          char=role_char.value,
                          user=role_usr.value)
        chatbot = [(None, chat_display_format(tmp))]
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
        tmp = text_format(role_char_first.value,
                          char=role_char.value,
                          user=role_usr.value)
        chatbot = [(None, chat_display_format(tmp))]
        tmp = chat_template(role_char.value, tmp)
        model.eval_t(tmp)  # 此内容随上下文增加将被丢弃

        # ========== 保存角色卡-缓存 ==========
        with open(setting_cache_path.value, 'wb') as f:
            pass
        tmp = model.save_session(setting_cache_path.value)
        print(f'save cache {tmp}')


# ========== 显示用户消息 ==========
def btn_submit_usr(message, history):
    # print('btn_submit_usr', message, history)
    if history is None:
        history = []
    return "", history + [[message, '']]


# ========== 模型流式响应 ==========
def btn_submit_bot(history, _n_keep, _n_discard,
                   _temperature, _repeat_penalty, _frequency_penalty,
                   _presence_penalty, _repeat_last_n, _top_k,
                   _top_p, _min_p, _typical_p,
                   _tfs_z, _mirostat_mode, _mirostat_eta,
                   _mirostat_tau, _usr, _char,
                   _rag, _max_tokens):
    if len(_rag) > 0:  # 需要临时注入的内容
        t_rag = chat_template('system', _rag)
        model.eval_t(t_rag, _n_keep, _n_discard)
    t_msg = history[-1][0]
    t_msg = chat_template(_usr, t_msg)
    model.eval_t(t_msg, _n_keep, _n_discard)
    t_bot = chat_template(_char)
    completion_tokens = []
    token = None
    for token in model.generate_t(
            tokens=t_bot,
            n_keep=_n_keep,
            n_discard=_n_discard,
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
        completion_tokens = []
        history[-1][1] += all_text
        yield history, model.n_tokens
        if len(t_bot) > _max_tokens:
            break
    history[-1][1] = chat_display_format(history[-1][1])
    yield history, model.n_tokens
    model.eval_t(chat_template.im_end_nl, _n_keep, _n_discard)
    t_bot.extend(chat_template.im_end_nl)
    if len(_rag) > 0:  # 响应完毕后清除注入的内容
        # history t_rag t_msg t_bot -> history t_msg t_bot
        n_discard = len(t_rag)
        n_keep = model.n_tokens - len(t_bot) - len(t_msg) - len(t_rag)
        model.kv_cache_seq_ltrim(n_keep, n_discard)
    yield history, model.n_tokens


# ========== 待实现 ==========
def btn_rag_(_rag, _msg):
    retn = ''
    return retn


# ========== 聊天页面 ==========
with gr.Blocks() as chatting:
    with gr.Row(equal_height=True):
        chatbot = gr.Chatbot(height='60vh', scale=2, value=chatbot,
                             avatar_images=(r'assets/user.png', r'assets/chatbot.webp'))
        with gr.Column(scale=1, elem_id="area"):
            rag = gr.Textbox(label='RAG', lines=2, max_lines=999,
                             show_copy_button=True, elem_id="RAG-area")
            vo = gr.Textbox(label='VO', lines=2, max_lines=999,
                            show_copy_button=True, elem_id="VO-area")
            s_n_tokens = gr.Number(value=model.n_tokens, label='n_tokens')
    msg = gr.Textbox(label='Prompt', lines=2)
    with gr.Row():
        btn_rag = gr.Button("RAG")
        btn_submit = gr.Button("Submit")
        btn_retry = gr.Button("Retry")
        btn_com1 = gr.Button("自定义1")
        btn_com2 = gr.Button("自定义2")
        btn_com3 = gr.Button("自定义3")

    btn_rag.click(fn=btn_rag_, outputs=rag,
                  inputs=[rag, msg])

    btn_submit.click(fn=btn_submit_usr, api_name="submit",
                     inputs=[msg, chatbot],
                     outputs=[msg, chatbot]).then(
        fn=btn_submit_bot,
        inputs=[chatbot, setting_n_keep, setting_n_discard,
                setting_temperature, setting_repeat_penalty, setting_frequency_penalty,
                setting_presence_penalty, setting_repeat_last_n, setting_top_k,
                setting_top_p, setting_min_p, setting_typical_p,
                setting_tfs_z, setting_mirostat_mode, setting_mirostat_eta,
                setting_mirostat_tau, role_usr, role_char,
                rag, setting_max_tokens],
        outputs=[chatbot, s_n_tokens]
    )

    # ========== 用于调试 ==========
    btn_com1.click(fn=lambda: model.str_detokenize(model._input_ids), outputs=rag)

# ========== 开始运行 ==========
demo = gr.TabbedInterface([chatting, setting, role],
                          ["聊天", "设置", '角色'],
                          css=custom_css)
gr.close_all()
demo.queue().launch(share=False)
