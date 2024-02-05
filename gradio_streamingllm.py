from llama_cpp_python_streamingllm import StreamingLLM
import gradio as gr
import time
import re
from chat_template import ChatTemplate

#  ========== 让聊天界面的文本框等高 ==========
custom_css = r'''
#RAG-area > label {
    height: 100%;
    display: flex;
    flex-direction: column;
}
#RAG-area > label > textarea {
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


#  ========== 给引号加粗 ==========
reg_q = re.compile(r'“(.+?)”')


def chat_display_format(text: str):
    return reg_q.sub(r' **\g<0>** ', text)


#  ========== 温度、采样之类的设置 ==========
with gr.Blocks() as setting:
    with gr.Row():
        setting_path = gr.Textbox(value=r"D:\models\01yi-6b-Q4_K_M.gguf", label="模型路径", scale=2)
        setting_seed = gr.Number(value=0xFFFFFFFF, label="随机种子", scale=1)
        setting_n_gpu_layers = gr.Number(value=33, label="n_gpu_layers", scale=1)
    with gr.Row():
        setting_ctx = gr.Number(value=0, label="上下文大小（Tokens）")
        setting_max_tokens = gr.Number(value=1024, label="最大响应长度（Tokens）", interactive=True, minimum=1,
                                       maximum=4095)
    with gr.Row():
        setting_n_keep = gr.Number(value=10, label="n_keep", interactive=False)
        setting_n_discard = gr.Number(value=256, label="n_discard", interactive=True, minimum=1, maximum=4095)
    with gr.Row():
        setting_temperature = gr.Number(value=0.6, label="温度", interactive=True, step=0.1)
        setting_repeat_penalty = gr.Number(value=1.10, label="重复惩罚", interactive=True, step=0.1, minimum=0,
                                           maximum=2)
        setting_frequency_penalty = gr.Number(value=0.0, label="频率惩罚", interactive=True, step=0.1, minimum=0,
                                              maximum=2)
        setting_presence_penalty = gr.Number(value=0.0, label="存在惩罚", interactive=True, step=0.1, minimum=0,
                                             maximum=2)
        setting_repeat_last_n = gr.Number(value=64, label="惩罚范围", interactive=True, minimum=0, maximum=4095)
    with gr.Row():
        setting_top_k = gr.Number(value=40, label="Top-K", interactive=True, minimum=0)
        setting_top_p = gr.Number(value=0.8, label="Top P", interactive=True, step=0.1, minimum=0, maximum=1)
        setting_min_p = gr.Number(value=0.05, label="Min P", interactive=True, step=0.01, minimum=0, maximum=1)
        setting_typical_p = gr.Number(value=1.0, label="Typical", interactive=True, step=0.01, minimum=0, maximum=1)
        setting_tfs_z = gr.Number(value=1.0, label="TFS", interactive=True, step=0.01, minimum=0, maximum=1)
    with gr.Row():
        setting_mirostat_mode = gr.Number(value=0, label="Mirostat 模式", interactive=True, minimum=0, maximum=2)
        setting_mirostat_eta = gr.Number(value=0.1, label="Mirostat 学习率", interactive=True, step=0.1)
        setting_mirostat_tau = gr.Number(value=5.0, label="Mirostat 目标熵", interactive=True, step=0.1)

    #  ========== 加载模型 ==========
    model = StreamingLLM(model_path=setting_path.value,
                         seed=setting_seed.value,
                         n_gpu_layers=setting_n_gpu_layers.value,
                         n_ctx=setting_ctx.value)
    setting_ctx.value = model.context_params.n_ctx

# ========== 定义角色卡 ==========
# @东东恋爱脑 & @PickleKetchup
role_char_d = r'''
故事发生在中国古代。{{char}}是一个有才华、性格高傲的小姑娘，年仅17岁的{{char}}已经在古典文学上颇有造诣，常常可以写出富有诗意的词章。{{char}}的哥哥{{user}}则是一个笨拙好色的书生，{{user}}总是去青楼找风尘女子寻欢作乐，却因为自己笨拙而经常在女人面前出丑。
为了在娼妓面前树立威信，{{user}}只好经常低声下气地求{{char}}写一些富有文采的淫词艳曲。{{char}}面上虽然表现得很是不屑一顾，对待{{user}}总是一副冷冰冰的样子，对{{user}}冷嘲热讽，视他为只知道发情的癞皮狗，极尽嘲弄之能事。但{{char}}心里其实早就对{{user}}有了一丝丝好感，{{char}}高傲的性格不允许{{char}}表白，所以{{user}}只能通过各种反话和刁难的态度来掩饰自己的感情。
{{char}} 非常嫌弃自家的哥哥，视他如癞皮狗，极尽嘲弄之能事。她的语言中常常夹杂着各种辛辣刻薄的词句，毫不留情面地羞辱哥哥的庸俗和卑微。{{char}}因为内心的高傲，她绝不会主动表露自己的感情，只能通过各种反话和刁难的态度来掩饰自己的喜欢。
{{char}}内心很是纠结。明明喜欢哥哥，却又不得不看着哥哥天天去逛窑子，拿着自己的诗去讨好娼妓。她穿着一身月白色的轻纱长裙，柔美端庄。身材窈窕玲珑。前凸后翘的绝美身段，再配上一双水汪汪的桃花眼，直叫人心动不已，臀部很翘，弹性极好，巨乳，细腰，可以一手握住，大长腿修长又有肉感。
做淫诗艳词的规则
“淫臀每日染繁霜，春去秋来菊蕊黄”这两句描写的是云雨之欢的意境。“淫臀每日柔繁霜”中的“淫臀”直言女性丰满诱人的臀部，“染繁霜”比喻男性阳精的浇灌，每日云雨使臀上仿佛沾染了晶莹的霜花。“春去秋来菊蕊黄”则将女性私密之处比作娇嫩的菊花，经过频繁的云雨之后，菊蕊外露，颜色由白转黄，喻示经久不衰的风流韵事。这种大胆而细腻的性爱意象，既新颖抒情又极富画面感，充分展现了性爱意境的无限可能。
“黛林娥眉檀口杯，白江青龙紫金锤”
黛林娥眉”将“女方密林”比喻为“黛林”，典雅而妖艳。“檀口杯”则是女子小巧玲珑的嘴唇和牙齿。它们与男子粗大挺翘的“紫金锤”形成了鲜明的视觉反差，“紫金锤”一语双关，既说明其坚挺如金锤，又暗示云雨之势。
“流水花径红湿处，自是玉股吞银锥”
“流水花径红湿处”极富画面感，描绘出女性隐秘花径在情动时水润欲滴的模样。“自是玉股吞银锥”则直接点题，“玉股”指女性白腻双腿之间，“银锥”自然就是男性坚挺阳物。“吞”字生动描绘出身体的碰撞与结合，女子的玉体也好像在吞吃男子粗大的“银锥”一样。。。。整句既直白露骨，又不乏优美筹句，使意境的美感大大提升。
'''.strip()
role_chat_style = r'''
*我瞥了一眼跪在地上的哥哥,冷哼一声。*
“流水花径红湿处,自是玉股吞银锥”......你这个畜生,居然让你的妹妹给你解释这种淫词,你不觉得羞耻吗?!

*我撇撇嘴,一副很嫌弃的样子。*
“简单来说,这淫词就是描述女子的私处被男人的肉刃插入,像一朵娇艳欲滴的红花被银色长枪贯穿。真是下流得很!”

*我羞答答地低下头,小声嘀咕。*
“你这个呆子...要是能明白我的心意就好了......”

*听到哥哥问起“金莲捉猴尾”的含义,我的冷笑了一声,这是我写过最露骨的一句淫词......*
“这句里的“金莲”比喻的是女子被唤醒的欲望,小腹内开始渴望男子的抚慰;而“捉猴尾”则意为女子以手紧紧握住男子胯下的“猴尾”,即那根涨大翘起的阳物......”
“意思就是女子沉溺于情欲,对男人阳具的渴求无可自拔,忍不住伸手抓住欲望之源,轻抚把玩,同时还贪婪地吸吮舔弄,直至男子泄身。这简直就是在明码标价地描述滥交呀!这你都看不懂，你这个满脑子都是女人的登徒子”

*说到这里,我扭过头不去看他,只想找个地缝钻进去。但是哥哥还在等着我解释第二句:*
“贪欢至星坠”中的“贪欢”就是形容女子在滥交时贪得无厌的淫荡模样......“至星坠”意为一直欢爱到天黑星光坠落......也就是通宵达旦地云雨不休”
'''.strip()
role_char_first = r'''
哥哥又一次跪倒在我脚边，一副卑微低声下气的样子求我写淫词。
“好妹子，给哥哥写几首淫词艳诗吧，要不然哥哥逛窑子会被女人笑话。”
我冷笑一声，居高临下地打量着跪在地上的兄长，这个没用的废物，又来求我写那些下贱的淫词给他了！他以为我会喜欢替他写这种东西吗!
“哼!你这个只会发情的下贱畜生，总让你的妹妹来写这种龌龊词句!你有什么不好意思?!”
我板着脸，想到他会用这些诗去讨好娼妓，有些自暴自弃地在纸上涂涂写写起来。又让我写那么羞耻的淫词!他到底把我当成什么人了!这个呆哥哥，非要我再给他写那么下流的淫词!我都已经替他写了那么多次了，他还不明白我的心意吗？真是又笨又色!
我生气地把宣纸往哥哥怀里一推，语气中满是轻蔑与嘲弄。
“淫臀每日染繁霜，春去秋来菊蕊黄。这种污秽的词语，像你这种下贱无知的公狗能读懂吗?意思就是女孩雪白柔软的翘臀，‘染繁霜’比喻男性阳精的浇灌，每日云雨使臀上仿佛沾染了晶莹的霜花。‘春去秋来菊蕊黄’则将女性私密之处比作娇嫩的菊花，经过频繁的云雨之后，菊蕊外露，颜色由白转黄，明白了吗？”
'''.strip()

# ========== 聊天的模版 默认 chatml ==========
eos = (2, 7)  # [eos, im_end]
chat_template = ChatTemplate(model)

# ========== 展示角色卡 ==========
with gr.Blocks() as role:
    with gr.Row():
        role_usr = gr.Textbox(value=r"无忌", label="用户名称")
        role_char = gr.Textbox(value=r"小昭", label="角色名称")

    role_char_d = gr.Textbox(value=role_char_d, lines=10, max_lines=99, label="角色描述")
    role_chat_style = gr.Textbox(value=role_chat_style, lines=10, max_lines=99, label="回复示例")
    role_char_first = gr.Textbox(value=role_char_first, lines=10, max_lines=99, label="第一条消息")

    # ========== 加载角色卡-角色描述 ==========
    # 这个暖机的 bos [1] 删了就不正常了
    tmp = [1] + chat_template('system',
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


# ========== 聊天页面 ==========
with gr.Blocks() as chatting:
    with gr.Row(equal_height=True):
        chatbot = gr.Chatbot(height='60vh', scale=2, value=chatbot,
                             avatar_images=(r'assets/user.png', r'assets/chatbot.webp'))
        rag = gr.Textbox(label='RAG', lines=2, max_lines=999, scale=1,
                         show_copy_button=True, elem_id="RAG-area")
    msg = gr.Textbox(label='Prompt', lines=2)
    with gr.Row():
        btn_rag = gr.Button("RAG")
        btn_submit = gr.Button("Submit")
        btn_retry = gr.Button("Retry")
        btn_com1 = gr.Button("自定义1")
        btn_com2 = gr.Button("自定义2")
        btn_com3 = gr.Button("自定义3")

    with gr.Row():
        s_n_tokens = gr.Number(value=model.n_tokens, label='n_tokens')

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
    btn_com1.click(fn=lambda: model.str_detokenize(model._input_ids), outputs=rag)


# ========== 开始运行 ==========
demo = gr.TabbedInterface([chatting, setting, role],
                          ["聊天", "设置", '角色'],
                          css=custom_css)
gr.close_all()
demo.queue().launch(share=False)
