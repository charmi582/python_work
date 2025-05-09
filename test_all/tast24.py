def a(n):
    b=[]
    for i in range(1, n+1):
        result=True
        s=str(i)
        for j in range(len(s)):
            if s[j]=='3' or s[j]=='7':
                result=False
                break
        else:
            b.append(i)
    return b
n=int(input())
print(a(n))
