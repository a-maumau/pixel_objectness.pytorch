import argparse
import os
from PIL import Image
import numpy as np
from tqdm import tqdm


def binarize(image_np):
    """
        binarize an image.
    """
    i, j = np.where(image_np > 0)
    image_np[i,j] = 1

    return image_np

def convert_binary_class(image_dir, output_dir):
    """
        binarize the images in 'image_dir' and save into 'output_dir'.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)
    
    for i, image in enumerate(tqdm(images)):
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as mask:
                open_mask = np.asarray(mask)
                open_mask.flags.writeable = True
                
                bin_mask = binarize(open_mask)
                
                bin_mask = Image.fromarray(np.uint8(bin_mask))
                bin_mask.save(os.path.join(output_dir, image), bin_mask.format)
        """
        if i % 100 == 0:
            print ("[%d/%d] Resized the images and saved into '%s'."
                   %(i, num_images, output_dir))
        """

def main(args):
    convert_binary_class(args.image_dir, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='./data/',
                        help='directory for train images')
    parser.add_argument('--output_dir', type=str, default='./bin_image/',
                        help='directory for saving resized images')

    args = parser.parse_args()
    main(args)