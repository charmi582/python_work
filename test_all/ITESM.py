a = int(input())
for i in range(a):
    b = int(input())
    c = 0 
    d = 0
    k = b 
    result = ''
    while b > 0:
        result = str(b % 2) + result
        b //= 2
    for j in result:
        if j == '1':
            c += 1
    print(c, end=' ')