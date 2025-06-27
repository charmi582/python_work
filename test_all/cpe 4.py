n = int(input())
from collections import defaultdict

group = defaultdict(lambda: defaultdict(int))

for _ in range(n):
    word = input()
    first = word[0]
    group[first][word] += 1

# 對每個開頭字母進行處理
for ch in sorted(group.keys()):
    words = group[ch]
    max_count = max(words.values())
    
    # 找到所有符合最大次數的字串
    candidates = [w for w in words if words[w] == max_count]
    result = min(candidates)  # 挑字典順序最小的
    print(f"{ch} {result}")
