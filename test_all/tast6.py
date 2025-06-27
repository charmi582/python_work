def check_password_strength(pw: str):
    a="!@#$%^&*"
    if len(pw)<6:
        return("簡單")
    for i in pw:
        for j in a:
            if i==j:
                return("難")
    return("正常")
pw=input()
print(check_password_strength(pw))
    