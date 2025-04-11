def a(n):
    b=[]
    for i in range(1, n+1):
        s=str(i)
        result=True
        c={}
        for j in s:
            if j in c:
                result=False
            
            c[j]=1
        if result:
            b.append(i)
    return b
n=int(input())
print(a(n))