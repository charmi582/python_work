def a(n):
    word=n.split()
    count={}
    for i in word:
        l=len(i)
        if l not in count:
            count[l]={}
        count[l][i]=count.get(i, 0)+1
        result={}
        for j in count:
            dict=count[j]
            max_count=max(dict, key=lambda k: dict[k])
            result[j]=max_count
    return result
n=input()
print(a(n))