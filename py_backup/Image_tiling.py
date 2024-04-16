from PIL import Image
import os
from tqdm import tqdm
import pandas as pd

df = pd.read_csv('train.csv')

image_files = df['upscale_img_path'].tolist()

output_dir = './train/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for img_path in tqdm(image_files, desc="이미지 처리 중"):
    img = Image.open(img_path)

    img_filename = os.path.basename(img_path)

    for i in range(4):
        for j in range(4):
            patch = img.crop((i * 64, j * 64, (i + 1) * 64, (j + 1) * 64))

            patch_filename = os.path.join(output_dir, f"{os.path.splitext(img_filename)[0]}_patch_{i*2+j+1}.png")
            patch.save(patch_filename)

print("패치 저장 완료.")
