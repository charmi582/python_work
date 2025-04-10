def a(n):
    b=[]
    for i in range(1, n+1):
        result=True
        s=str(i)
        for j in range(1, len(s)):
            if int(s[j-1])+int(s[j])==10:
                result=False
                break
        if result:
            b.append(i)
    return b
n=int(input())
print(a(n))
