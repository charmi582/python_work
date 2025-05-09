def a(n):
    b=[]
    for i in range(1, n):
        if '4' in str(i):
            continue
        else:
            b.append(i)
    return b
n=int(input())
print(a(n))