def a(n):
    count={}
    word=n.split()
    for j in word:
            if j in count:
                count[j]+=1
            else:
                count[j]=1
    return count
n=input()
print(a(n))
 
        