def a(n):
    count={str(i): 0 for i in range(10)} 
    for i in str(n):
        count[i]+=1
    max_count=max(count.values())
    result=[k for k, v in count.items() if v==max_count]
    return result
n=int(input())
print(a(n))