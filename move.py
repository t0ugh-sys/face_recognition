# -*- coding:utf-8 -*-
# @Author  : t0ugh
# @Date    : 2025/6/16 17:09
# @Description: 
# @Version : v1
import os
import shutil


def move_photos(source_dir, target_dir):
    """将所有子文件夹中的图片移动到目标文件夹，自动处理重名冲突"""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"创建目标文件夹: {target_dir}")

    # 支持的图片格式列表（可扩展）
    image_exts = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
    moved_count = 0

    # 递归遍历所有子文件夹
    for root, _, files in os.walk(source_dir):
        for filename in files:
            if filename.lower().endswith(image_exts):
                src_path = os.path.join(root, filename)
                dst_path = os.path.join(target_dir, filename)

                # 处理重名文件：自动添加序号 (file.jpg → file_1.jpg)
                counter = 1
                while os.path.exists(dst_path):
                    base, ext = os.path.splitext(filename)
                    dst_path = os.path.join(target_dir, f"{base}_{counter}{ext}")
                    counter += 1

                shutil.move(src_path, dst_path)
                print(f"移动: {src_path} → {dst_path}")
                moved_count += 1

    print(f"操作完成！共移动 {moved_count} 张图片。")


# 配置路径（修改为您的实际路径）
source_folder = r"D:\datasets\FACE"  # 照片散落的根目录
target_folder = r"D:\datasets\FACE"  # 目标集中文件夹

move_photos(source_folder, target_folder)