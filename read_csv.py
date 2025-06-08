import pandas as pd

# 1. 读取文件（假设是CSV，以空格分隔）
df = pd.read_csv('C:/Users/12531/Desktop/BIGC/COLOR/ISOStandards/Application24790_Amd/Application24790_Amd/OECFfile/image2-1.csv', sep='\s+', header=None)  # \s+ 匹配多个空格

# 2. 检查第一列是否需要拆分
if df.shape[1] == 1:  # 如果只有一列
    # 按空格拆分，并扩展成三列
    split_data = df[0].str.split(expand=True)
    # 只保留前三列（防止有多余的空格）
    df = split_data.iloc[:, :3]  # 取前3列

# 3. 保存结果（可选）
df.to_csv('image2-1-oecf.csv', index=False, header=False)

print("分割后的数据：")
print(df.head())