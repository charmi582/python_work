def a(n):
    word=n.split()
    count={}
    for i in word:
        count[i]=count.get(i, 0)+1
    max_count=max(count.items(), key=lambda x:x[1])
    return max_count[0]
n=input()
print(a(n))
    