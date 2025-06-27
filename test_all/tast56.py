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
        max_count=max(dict, key=lambda k:dict[k])
        result=max_count
    return result
n=input()
print(a(n))
