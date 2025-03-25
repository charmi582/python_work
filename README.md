#python_work
##簡介
python 列車制換員

##功能
使車廂以正瘸排序排列

##程式碼
a=eval(input())
g=0
for j in range(a):
    b=eval(input())
    d=str(input())
    for i in range(len(d)-1):
            if d[i] > d[i + 1]:
                c = d[i + 1]
                d = d[:i] + c + d[i] + d[i+2:] 
                g+=1
    print(f"Optimal train swapping take {g} swaps.")
第9行輸入a可以輸入幾次
第十行g=0為了運算最後swap了多少次
11-14行使用兩個for迴圈以便使用氣泡排序法
第12行輸入b表示有多少節車廂
第13行輸入d字串表示車廂號碼
第15行-第17行使用if判斷式使兩數交換
最後print出總共交換了多少次車廂
