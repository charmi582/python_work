def a(n):
    b=''
    for i in range(n+1):
        c=str(i)
        p=0
        for j in c:
            p+=int(j)
        if p==7:
                b+=c+" "
    return b
n=int(input())
print(a(n))
            