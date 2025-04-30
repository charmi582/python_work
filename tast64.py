def a(n):
    word=n.split()
    count={}
    for i in word:
        first=i[0]
        if first not in count:
            count[first]={}
        count[first][i]=count[first].get(i, 0)+1
    sorted_count=sorted(count.items(), key=lambda x: [x], reverse=False)
    return sorted_count
n=input()
print(a(n))