with open('./train', 'r', encoding='utf-8') as r:
    s = r.read()
    c1 = s.count('，')
    c2 = s.count('。')
    c3 = s.count('？')
    c4 = s.count('！')
    c5 = s.count('：')
    c6 = s.count(' ')
    c = c1 + c2 + c3 + c4 + c5 + c6
    c1 = c1/c
    c2 = c2/c
    c3 = c3/c
    c4 = c4/c
    c5 = c5/c
    c6 = c6/c
    print(1/c1)
    print(1/c2)
    print(1/c3)
    print(1/c4)
    print(1/c5)
    print(1/c6)
    # 15, 24, 243, 117, 269, 1
