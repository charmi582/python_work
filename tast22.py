def a(n):
    b=[]
    for i in range(1, n+1):
        c=0
        for j in str(i):
            c+=int(j)
        for z in range(2, c):
            if c%z==0:
                break
        else:
            b.append(i)
    return b
n=int(input())
print(a(n))