def a(n):
    c=''
    d='7'
    for i in range(n+1):
        b=str(i)
        for j in b:
            if j==d:
                c+=b+' '
    return c
n=int(input())
print(a(n))