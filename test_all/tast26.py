def a(n):
    b=[]
    for i in range(1, n+1):
        count={}
        s=str(i)
        result=True
        for j in range(1, len(s)):
            if s[j] in count:
                result=False   
            count[s[j]]=1
        if result:
            for z in range(1, len(s)):
                g=int(s[z-1])+int(s[z])
                b.append(g)
    return b
n=int(input())
print(a(n))