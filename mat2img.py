"""
    for VOC and SBD dataset
"""
import scipy.io
import os

img_file_extension = '.png'
mat_file_extension = '.mat'

def read_mat(mat_filename, key='GTcls'):
        mat = scipy.io.loadmat(mat_filename, mat_dtype=True, squeeze_me=True, struct_as_record=False)
        return mat[key].Segmentation

def convert_mat2img(image_list, mat_dir="cls", output_dir="cls_png"):
    """
        image_list should be a list of string of file name without extension.
        image_list = ["2008_0000002", ...]
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_name in image_list:
        mat = read_mat(os.path.join(mat_dir, img_name+mat_file_extension), key='GTcls'):
        io.imsave(os.path.join(output_dir, img_name+img_file_extension), mat)
        print("converted {}".dormat(img_name))

def read_from_txt(txt_file="train.txt"):
    with open(os.path.join(VOC_list), "r") as f:
        image_names = f.readlines()
    
    image_names = [img_name.rstrip("\n") for img_name in image_names]
    return image_names

def read_from_currentdir():
    dir_files = os.listdir()
    dir_files = [file.rstrip(mat_file_extension+"\n") for file in dir_files]

    return dir_files

def main(args):
    if args.here:
        image_list = read_from_currentdir()
    else:
        image_list = read_from_txt(args.image_list)

    convert_mat2img(image_list, args.mat_dir, args.output_dir)

if __name__ == '__main__':
        parser = argparse.ArgumentParser()

        parser.add_argument('--image_list', type=str, default='./dataset/train.txt', help='dir of the image list.')
        parser.add_argument('--mat_dir', type=str, default='./dataset/cls', help='mat dir.')
        parser.add_argument('--output_dir', type=str, default='./dataset/cls_png', help='output dir.')
                
        # flags
        parser.add_argument('-here', action="store_true", default=False, help='read from current directory')

        args = parser.parse_args()
        
        main(args)