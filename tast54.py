def a(n):
    word=n.split()
    count={}
    for i in word:
        first=i[0]
        if first not in count:
            count[first]={}
        count[first][i]=count[first].get(i, 0)+1
    return count
n=input()
print(a(n))