a = [1, 2]
while a[-1] < 100000000:
    a.append(a[-1] + a[-2])
# 讀入要轉換的數量
N = int(input())
for case in range(N):
    num = int(input())
    print(f"{num} = ", end='')

    found = 0  # 控制是否已經找到第一個 1（以避免前導 0）
    
    for f in a[::-1]:  # 從大的費氏數開始往下檢查
        if num >= f:   # 如果可以用這個數
            num -= f   # 減去它
            found = 1  # 設定已經開始輸出
            print(1, end='')
        elif found:    # 若已開始輸出，就補 0
            print(0, end='')

    print(" (fib)")