import copy


class ChatTemplate:
    cache = {}
    roles = set()

    def __init__(self, model, im_start=r'<|im_start|>', im_end=r'<|im_end|>', nl='\n'):
        self.model = model
        self.nl = nl
        self.im_start = im_start
        self.im_start_token = model.tokenize(self.im_start.encode('utf-8'), add_bos=False, special=True)
        self.im_end = im_end
        self.im_end_nl = model.tokenize((self.im_end + self.nl).encode('utf-8'), add_bos=False, special=True)
        self.eos = [model._token_eos, self.im_end_nl[0]]
        self.onenl = [self.im_end_nl[-1]]
        tmp = model.tokenize(('\r' + self.nl).encode('utf-8'), add_bos=False, special=True)
        if len(tmp) == 1:
            self.onenl.append(tmp[0])
        self.onerl = model.tokenize(b'\r', add_bos=False, special=True)
        self.nlnl = None
        tmp = model.tokenize((self.nl + self.nl).encode('utf-8'), add_bos=False, special=True)
        if len(tmp) == 1:
            self.nlnl = tmp[0]
        print('ChatTemplate', self.eos, self.im_end_nl, self.onerl, self.onenl, self.nlnl)

    def _get(self, key: str):
        if key in self.cache:
            return copy.deepcopy(self.cache[key])  # 深拷贝一下
        else:
            value = self.model.tokenize((self.im_start + key + self.nl).encode('utf-8'), add_bos=False, special=True)
            self.cache[key] = copy.deepcopy(value)  # 深拷贝一下
            return value

    def _add_role(self, _role):
        if _role:
            self.roles.add('\n' + _role)

    def eos_in_role(self, history: str, t_bot):
        if not (history.endswith('\n') or history.endswith('\r')):
            return 0
        tmp = history.rstrip()
        for _role in self.roles:
            if tmp.endswith(_role):
                n = len(t_bot)
                for i in range(1, n):  # 找出需要弃置的tokens长度
                    tmp = self.model.str_detokenize(t_bot[n - i:])
                    if tmp.rstrip().endswith(_role):
                        print('eos_in_role', t_bot[n - i:], repr(tmp))
                        return i
                print('eos_in_role missing')
                break
        return 0

    def eos_in_nlnl(self, history: str, t_bot):
        if not (history.endswith('\n\n') or history.endswith('\n\r\n')):
            return 0
        n = len(t_bot)
        for i in range(1, n):  # 找出需要弃置的tokens长度
            tmp = self.model.str_detokenize(t_bot[n - i:])
            if tmp.endswith('\n\n') or tmp.endswith('\n\r\n'):
                if tmp.startswith(']'):  # 避免误判
                    return 0
                print('eos_in_nlnl', t_bot[n - i:], repr(tmp))
                return i
        print('eos_in_nlnl missing')
        return 0

    def __call__(self, _role, prompt=None):
        self._add_role(_role)
        if prompt is None:
            return self._get(_role)
        # print(_role, prompt, self.cache)
        prompt = self.im_start + _role + self.nl + prompt
        prompt = self.model.tokenize(prompt.encode('utf-8'), add_bos=False, special=True) + self.im_end_nl
        # print(self.model.str_detokenize(prompt), prompt)
        return prompt
