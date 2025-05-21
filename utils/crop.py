import os
from PIL import Image


def crop_images(src_dir, dest_dir, crop_rect):
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

        # 遍历源文件夹中的所有文件
    for filename in os.listdir(src_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # 你可以根据需要添加更多图像格式
            img_path = os.path.join(src_dir, filename)
            try:
                # 打开图像并检查分辨率
                img = Image.open(img_path)
                if img.size == (480, 480):  # 检查分辨率是否为480x480
                    # 裁剪图像
                    cropped_img = img.crop(crop_rect)
                    # 保存裁剪后的图像到目标文件夹
                    cropped_filename = os.path.join(dest_dir, filename)
                    cropped_img.save(cropped_filename)
                    print(f"Processed and saved: {cropped_filename}")
                else:
                    print(f"Skipped: {img_path} (resolution is not 480x480)")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

            # 设置源文件夹、目标文件夹和裁剪区域（左、上、右、下）


src_dir = r'../result/version1'
dest_dir = r'../result/version1/crop'
crop_rect = (100, 350, 225, 475) 

# 调用函数开始处理
crop_images(src_dir, dest_dir, crop_rect)
