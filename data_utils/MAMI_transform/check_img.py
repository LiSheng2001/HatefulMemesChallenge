# 检查哪些id重复了
import pandas as pd
import json

# 加载jsonl数据
def read_jsonl_to_dataFrame(jsonl_file: str):
    with open(jsonl_file) as f:
        lines = f.read().splitlines()
    line_dicts = [json.loads(line) for line in lines]
    return pd.DataFrame(line_dicts)

train = read_jsonl_to_dataFrame("/root/autodl-tmp/MAMI/train.jsonl")
dev = read_jsonl_to_dataFrame("/root/autodl-tmp/MAMI/dev.jsonl")
test = read_jsonl_to_dataFrame("/root/autodl-tmp/MAMI/test.jsonl")

# 似乎test和train里面id有重复的
a = set(train.id.values)
c = set(test.id.values)
print(c.intersection(a)) # 手动消除冲突了

print(1234)