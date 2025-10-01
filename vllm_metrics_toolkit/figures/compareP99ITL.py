import matplotlib.pyplot as plt
import numpy as np

# 数据
rps = [1, 5, 10]

# 不带 encoding 的数据 (ms 转换成秒方便对比)
no_encoding = [46.48/1000, 48.54/1000, 49.03/1000]
with_encoding = [106.88/1000, 149.88/1000, 146.61/1000]

x = np.arange(len(rps))  # x轴刻度位置
width = 0.35  # 柱子宽度

fig, ax = plt.subplots(figsize=(6,4))

# 画柱状图
rects1 = ax.bar(x - width/2, no_encoding, width, label='No Encoding', hatch="//")
rects2 = ax.bar(x + width/2, with_encoding, width, label='With Encoding', color="skyblue")

# 添加文字和标签
ax.set_ylabel('P99 ITL Latency (seconds)')
ax.set_xlabel('Queries per Second (RPS)')
ax.set_title('High Tail Latency (P99)')
ax.set_xticks(x)
ax.set_xticklabels(rps)
ax.legend()

plt.tight_layout()
plt.show()

plt.savefig('compareP99ITL.png')