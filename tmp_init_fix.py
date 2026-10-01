import re

files = [
    'ZQCNN/ZQ_CNN_MTCNN.h',
    'ZQCNN/ZQ_CNN_MTCNN_Interface.h',
    'ZQCNN/ZQ_CNN_MTCNN_NCHWC.h',
    'ZQCNN/ZQ_CNN_MTCNN_AspectRatio.h',
]

total = 0
for p in files:
    raw = open(p, 'rb').read().decode('utf-8')
    NL = '\r\n' if '\r\n' in raw else '\n'
    # 在 "if (!ret) { ...clear... } else ..." 之后、show_debug_info 之前,
    # 失败时直接 return, 避免对已 clear 的 vector 取 [0]
    pat = re.compile(
        r'(?P<pre>(?:\t+)if \(!ret\)\s*\n'
        r'(?:\t*)\{\s*\n'
        r'(?:(?:\t| )[^\n]*\n)*?'
        r'(?:\t*)\}\s*\n'
        r'(?:\t+)else\s*\n'
        r'(?:\t+)[^\n]*\n)'
    )
    out = []
    pos = 0
    n = 0
    for m in pat.finditer(raw):
        # 找到紧跟其后的 if (show_debug_info)
        tail = raw[m.end():m.end() + 200]
        if not tail.lstrip().startswith('if (show_debug_info)'):
            continue
        out.append(raw[pos:m.end()])
        out.append(m.group('pre')[:m.group('pre').rindex('\n')] + '\n')
        out.insert(len(out) - 1, '\t\t\tif (!ret)\n\t\t\t\treturn false;\n')
        pos = m.end()
        n += 1
    if n:
        out.append(raw[pos:])
        open(p, 'wb').write(''.join(out).encode('utf-8'))
    print(p, 'patched', n)
    total += n
print('total', total)
