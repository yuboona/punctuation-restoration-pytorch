with open('./test_', encoding='utf-8') as f:
    a = f.readlines()
    res = []
    for s in a:
        for i in range(len(s)):
            res.append(s[i])

    with open('./test', 'w',encoding='utf-8') as w:
        for i in res:
            w.write(i)
            w.write(' ')
