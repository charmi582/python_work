## python work

## 題目
在這個問題中你必須將數列文字往順時針方向旋轉90度。也就是說將原本由左到右，由上到下的句子輸出成由上到下，由右到左。

輸入說明：輸入最多不會超過100列，每列最多不會超過100個字元。 合法的字元包括：換行，空白，所有的標點符號，數字，以及大小寫字母。（注意：Tabs並不算是合法字元。） 最後一列輸入必須垂直輸出在最左邊一行，輸入的第一列必須垂直輸出在最右邊一行。 請參考sample intput/output。

## 程式碼``
 ``` python
lines = [] 
while True:
    line = input()
    if line == "":
        break
    lines.append(line)
max_len = max(len(line) for line in lines)
lines = [line.ljust(max_len) for line in lines]
linepro = []
for col in range(max_len):
    new_line = ""
    for row in range(len(lines)):
        new_line += lines[row][col]
    linepro.append(new_line)
for line in linrpro:
    print(line)
 ``` 
    
## 講解
11-14行為輸入字串當line是空白時，跳出字串。
15行為將我輸入的字串加入lines中
16行是網路找到尋找最常字串的方法
17行也是網路上找到的方法將其餘比較小的字串後方自動補空格
18行為我新建立一個矩陣
19-23行為將平行轉換成垂直
24-25行為print出我的答案

## 遇到問題
輸出的答案跟題目給的答案是相反的
