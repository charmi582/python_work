def a(n):
    count={}
    for i in str(n):
        if i in count:
            count[i]+=1
        else:
            count[i]=1
    min_count=min(count.values())
    result=[k for k, v in count.items() if v==min_count]
    return result
n=int(input())
print(a(n))