n=input()
count={}
b=0
for i in n:
    if i not in count:
        count[i]=0
        b+=1
    count[i]=count.get(i, 0)+1
print(f"不同的字元有:{b}種")
print(count)