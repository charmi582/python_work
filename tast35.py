def a(n):
    count={}
    for i in n.lower():
        if i.isalpha():
            if i in count:
                count[i]+=1
            else:
                count[i]=1
    return count
n=input()
print(a(n))