def a(n):
    word=n.split()
    count={}
    for i in word:
        count[i]=count.get(i, 0)+1
    min_count=min(count.keys(), key=lambda x: count[x])
    return min_count
n=input()
print(a(n))