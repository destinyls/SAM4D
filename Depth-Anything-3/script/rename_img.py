import os

path = '/home/yanglei/Depth-Anything-3/images/v4_save'
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
files.sort()  # 可替换为其他排序逻辑

for i, filename in enumerate(files):
    old_path = os.path.join(path, filename)
    new_path = os.path.join(path, (str(i)+'.jpg'))
    os.rename(old_path, new_path)
