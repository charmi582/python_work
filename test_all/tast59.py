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
        max_count=max(dict.values())
        result=[k for k, v in dict.items() if v==max_count]
        z[j]=result
    return z
n=input()
print(a(n))