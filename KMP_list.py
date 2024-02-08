def compute_lps_array(sublist):
    """
    计算模式串的最长前缀后缀匹配数组（LPS数组）
    """
    lps = [0] * len(sublist)
    j = 0
    i = 1
    while i < len(sublist):
        if sublist[i] == sublist[j]:
            j += 1
            lps[i] = j
            i += 1
        else:
            if j != 0:
                j = lps[j - 1]
            else:
                lps[i] = 0
                i += 1
    return lps


def kmp_search(main_list, sublist, _start=0, _end=None, lps=None):
    """
    使用KMP算法在列表上查找子串
    """
    if not sublist:
        return 0
    if _end is None:
        _end = len(main_list)
    if lps is None:
        lps = compute_lps_array(sublist)
    i = _start  # 指向主串的索引
    j = 0  # 指向子串的索引
    while i < _end:
        if main_list[i] == sublist[j]:
            i += 1
            j += 1
            if j == len(sublist):
                return i - j
        else:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return -1


if __name__ == '__main__':
    a = [1, 1, 3, 2, 3, 6, 7, 8, 3, 2, 3]
    b = [3, 2, 3]
    c = compute_lps_array(b)
    print(kmp_search(a, b, lps=c))
    print(kmp_search(a, b, 3, lps=c))
    print(kmp_search(a, b, 3, 10, lps=c))
    print(kmp_search(a, b, 9, lps=c))
