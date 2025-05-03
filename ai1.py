a=int(input("請輸入你的平均速度:"))
b=int(input("請輸入我的震動次數"))
if a<=5 and b<3:
    print("步行")
elif a<25 and b>5:
    print("機車")
elif a>=25 and b<=3:
    print("汽車")
