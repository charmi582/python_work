def a(n):
    count={}
    for i in n:
        if i in count:
            count[i]+=1
        else:
            count[i]=1
    max_count=max(count.values())
    result=[k for k, v in count.items() if v==max_count]
    return result
n=input()
print(a(n))