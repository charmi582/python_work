def a(n):
    word=n.split()
    count={}
    for i in word:
        count[i]=count.get(i, 0)+1
    count_max=max(count.values())
    result=[k for k, v in count.items() if v==count_max]
    return result
n=input()
print(a(n))