def a(n):
    b=[]
    for i in range(1, n+1):
        result=True
        for j in str(i):
            d=int(j)
            if d%2==0:
                result=False
        if result:
            b.append(i)
    return b
n=int(input())
print(a(n))
