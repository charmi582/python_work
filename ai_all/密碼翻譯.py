n=int(input())
count={}
for i in range(n):
   s=input().strip()
   for w in s:
      if w.isalpha(): 
         w = w.upper()
         count[w]=count.get(w, 0)+1
f=sorted(count,key=lambda x:(-count[x],x))
for i in f:
  print(f"{i} {count[i]}")