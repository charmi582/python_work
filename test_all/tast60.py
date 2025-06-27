def a(n):
    word=n.split()
    count={}
    for i in word:
        long=len(i)
        if long not in count:
            count[long]={}
        count[long][i]=count[long].get(i, 0)+1
    z={}
    for j in count:
        dict=count[j]
        total=sum(dict.values())
        z[j]=total
    max_count=max(z.values())
    result=[k for k, v in z.items() if v==max_count]

    return result
n=input()
print(a(n))
