def a(n):
    b=[]
    
    for i in range(1, n+1):
        result=True
        c=0
        for j in str(i):
            c+=int(j)
        if c==10:
            result=False
        if result:
            b.append(i)
    return b
n=int(input())
print(a(n))
