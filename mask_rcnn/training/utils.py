import os
import glob
import random

# Utils made the datasets to be able to get randomized. 
# SO the models can have random orders and preventing "understanding" the same data if the original name is been categorized.
def randomize_datasets(images_dir, annots_dir):
    # Get the lists of both images and annots
    image_list = glob.glob(os.path.join(images_dir, '*.jpg'))
    annots_list = glob.glob(os.path.join(annots_dir, '*.xml'))
    
    random.Random(42).shuffle(image_list)
    random.Random(42).shuffle(annots_list)

    padding = len(str(len(image_list)))

    # Randomize the name so later the dataset can be vary
    for n, filepath in enumerate(image_list, 1):
        os.rename(filepath, os.path.join(images_dir, '{:>0{}}.jpg'.format(n, padding)))

    for n, filepath in enumerate(annots_list, 1):
        os.rename(filepath, os.path.join(annots_dir, '{:>0{}}.xml'.format(n, padding)))
        
if __name__ == '__main__':
    randomize_datasets('datasets/images', 'datasets/annots')