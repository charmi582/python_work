def a(n):
    result = []
    for i in range(1, n + 1):
        all_odd = True  # 旗標，預設為全部都是奇數
        for ch in str(i):
            if int(ch) % 2 == 0:
                all_odd = False
                break  # 有偶數就跳出，不用看了
        if all_odd:
            result.append(i)
    return result

n = int(input())
print(a(n))