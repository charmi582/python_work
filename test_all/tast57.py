def a(n):
    word=n.split()
    count={}
    for i in word:
        last=i[-1]
        if last not in count:
            count[last]={}
        count[last][i]=count[last].get(i, 0)+1
    result={}
    for j in count:
        dict=count[j]
        max_count=max(dict.values())
        z=[k for k, v in dict.items() if v==max_count]
        result[j]=z
    return z
n=input()
print(a(n))