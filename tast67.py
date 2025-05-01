def a(n):
    word=n.split()
    count={}
    for i in word:
        last=i[-1]
        if last not in count:
            count[last]={}
        count[last][i]=count[last].get(i, 0)+1
    z={}
    for j in count:
        dic=count[j]
        min_count=min(dic.values())
        result=[k for k, v in dic.items() if v==min_count]
        z[j]=result
    return z
n=input()
print(a(n))