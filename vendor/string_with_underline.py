def str_w_undln(text, start_pos, end_pos):
    result = ''

    start = max(text.rfind('\n', 0, start_pos.idx), 0)
    end = text.find('\n', start+1)
    if end < 0: end = len(text)

    line = text[start:end]
    col_start = start_pos.col
    col_end = end_pos.col

    result += line + '\n'
    result += (' ' * col_start) + ('~' * (col_end - col_start))

    return result.replace('\t', '')