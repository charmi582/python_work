def a(n):
    word=n.split()
    count={}
    for i in word:
        count[i]=count.get(i, 0)+1
    result=[k for k, v in count.items() if v>2]
    return result
n=input()
print(a(n))