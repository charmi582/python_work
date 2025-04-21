def a(n):
    word=n.split()
    count={}
    for i in word:
        count[i]=count.get(i, 0)+1
    count_sorted=sorted(count.items(), key=lambda x:x[1], reverse=True)
    result=[k for k, v in count_sorted[:3]]
    return result
n=input()
print(a(n))