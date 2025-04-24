# python work cryptanalysis
## 題目
密碼翻譯（cryptanalysis）是指把某個人寫的密文（cryptographic writing）加以分解。這個程序通常會對密文訊息做統計分析。你的任務就是寫一個程式來對密文作簡單的分析。

範例輸入 #1
3
This is a test.
Count me 1 2 3 4 5.
Wow!!!! Is this question easy?

範例輸出 #1
S 7
T 6
I 5
E 4
O 3
A 2
H 2
N 2
U 2
W 2
C 1
M 1
Q 1
Y 1
## 程式碼
```python=
n=int(input())
count={}
for i in range(n):
   s=input().strip()
   for w in s:
      if w.isalpha(): 
         w = w.upper()
         count[w]=count.get(w, 0)+1
f=sorted(dict,key=lambda x:(-dict[x],x))
for i in f:
  print(f"{i} {dict[i]}")
```
## 講解
一開始我有點看不太懂這個程式是怎麼去翻譯這個密碼，所以有上網找cpe的題目。
我看了網路上面的完整題目之後，發現這是基本的字典應用，那首先我先輸入一個整數n，然後建立一個字典count，建立一個for迴圈