def a(n):
    c=[]
    for i in range(1, n+1):
        result=True
        s=str(i)
        for j in range(1, len(s)):
            if s[j-1]==s[j]:
                result=False
        if result:
            c.append(i)
    return c
n=int(input())
print(a(n))