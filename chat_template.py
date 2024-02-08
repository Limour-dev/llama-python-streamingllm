import copy


class ChatTemplate:
    cache = {}

    def __init__(self, model, im_start=r'<|im_start|>', im_end=r'<|im_end|>', nl='\n'):
        self.model = model
        self.nl = nl
        self.im_start = im_start
        self.im_start_token = model.tokenize(self.im_start.encode('utf-8'), add_bos=False, special=True)
        self.im_end = im_end
        self.im_end_nl = model.tokenize((self.im_end + self.nl).encode('utf-8'), add_bos=False, special=True)

    def _get(self, key: str):
        if key in self.cache:
            return copy.deepcopy(self.cache[key])  # 深拷贝一下
        else:
            value = self.model.tokenize((self.im_start + key + self.nl).encode('utf-8'), add_bos=False, special=True)
            self.cache[key] = copy.deepcopy(value)  # 深拷贝一下
            return value

    def __call__(self, _role, prompt=None):
        if prompt is None:
            return self._get(_role)
        # print(_role, prompt, self.cache)
        prompt = self.im_start + _role + self.nl + prompt
        prompt = self.model.tokenize(prompt.encode('utf-8'), add_bos=False, special=True) + self.im_end_nl
        # print(self.model.str_detokenize(prompt), prompt)
        return prompt
