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
        dict_count=count[j]
        sorted_count=sorted(dict_count.items(), key=lambda x:x[1], reverse=True)
        z[j]=sorted_count
    return z
n=input()
print(a(n))