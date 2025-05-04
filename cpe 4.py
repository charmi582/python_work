n=int(input())
count={}
for i in range(n):
    a=input()
    for j in a:
        first=j[0]
    if a not in count:
        count[first]={}
    count[first][a]=count[first].get(i, 0)+1
    max_count=max(count[first].items())
    result=[k for k ,v in count[first].items() if v == max_count]
print( result)
        
