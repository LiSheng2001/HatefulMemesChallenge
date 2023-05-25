import glob

# 获取所有png
a = glob.glob("/root/autodl-tmp/MAMI/img_mask_3px/*png")

print(len(a)) # 22000正常