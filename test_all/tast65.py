def a(n):
    word=n.split()
    count={}
    for i in word:
        last=i[-1]
        if last not in count:
            count[last]={}
        count[last][i]=count[last].get(i, 0)+1
    sorted_count=sorted(count.items(), key=lambda x:x[0])
    result = dict(sorted_count)
    return result
n=input()
print(a(n))