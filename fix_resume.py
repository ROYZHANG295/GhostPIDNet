file_path = "/home/tstone10/miniconda3_py38/envs/openmmlab/lib/python3.8/site-packages/mmengine/runner/loops.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 用 if False 巧妙地屏蔽掉那个耗时的 for 循环，保持缩进不变
old_str = "for _ in range(skip):"
new_str = "if False:  # [已修复] for _ in range(skip):"

if old_str in content:
    content = content.replace(old_str, new_str)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("\n✅ 修复成功！导致卡死的 DataLoader 空转代码已被自动屏蔽。\n")
else:
    print("\n⚠️ 未找到目标代码，可能已经被修改过了。\n")
