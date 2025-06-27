def a(n):
    word=n.split()
    count={}
    for i in word:
        count[i]=count.get(i, 0)+1
    max_count=max(count.keys(), key=lambda x:count[x])
    return max_count
n=input()
print(a(n))