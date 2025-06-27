def a(n):
    count={}
    for i in n:
        if i in count:
            count[i]+=1
        else:
            count[i]=1
    sorted_count=sorted(count.items(),key=lambda x: x[1], reverse=True)
    result=[k for k, v in sorted_count[:3]]
    return result
n=input()
print(a(n))