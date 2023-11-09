import os
import argparse
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Resize images in a directory')

parser.add_argument('--input-dir', type=str, help='path to input directory')
parser.add_argument('--output-dir', type=str, help='path to output directory')
parser.add_argument('--size', type=int, help='size to resize images')

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

for filename in tqdm(os.listdir(args.input_dir)):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        with Image.open(os.path.join(args.input_dir, filename)) as img:
            img_resized = img.resize((args.size, args.size))
            img_resized.save(os.path.join(args.output_dir, filename))
