def a(n):
    word=n.split()
    count={}
    for i in word:
        count[i]=count.get(i, 0)+1
    sorted_item=sorted(count.items(), key=lambda x:x[1], reverse=True)
    result=[k for k, v in sorted_item[:2]]
    return result
n=input()
print(a(n))
