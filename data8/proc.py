import pkuseg

def one_char():
    """make txt to one character sequence.
    """
    with open('./test_', encoding='utf-8') as f:
        a = f.readlines()
        res = []
        for s in a:
            for i in range(len(s)):
                res.append(s[i])

        with open('./test', 'w', encoding='utf-8') as w:
            for i in res:
                w.write(i)
                w.write(' ')


def word_seg():
    with open('./test_', 'r', encoding='UTF-8') as r:
        content = r.read()
    # print(type(content))
    seg = pkuseg.pkuseg()

    cut = seg.cut(content)
    print(cut)

    with open('./test', 'w', encoding='utf-8') as w:
        for c in cut:
            if c == '，':
                continue
            if c == '。':
                continue
            if c == '？':
                continue
            if c == '！':
                continue
            if c == '：':
                continue
            w.write(c)
            w.write(' ')


word_seg()