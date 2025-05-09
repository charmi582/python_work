def a(n):
    count={}
    for i in n:
        if i in count:
            count[i]+=1
        else:
            count[i]=1
    max_count=max(count.items(), key=lambda x:x[1])[0]
    return max_count
n=input()
print(a(n))
