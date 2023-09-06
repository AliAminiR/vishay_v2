import imgaug.augmenters as iaa
import os
import imageio.v2 as imageio
import gc
import cv2


def main():
    for i in range(0, 20):
        input_dir = r"C:\Users\A750290\Projects\Vishay\Data\cut\Input_cleansed\val\valid"
        output_dir = r"C:\Users\A750290\Projects\Vishay\Classification\input\val\valid"

        seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.TranslateX(px=(-50, 50)),
            iaa.TranslateY(px=(-50, 50)),
            # iaa.LinearContrast((0.75, 1.5)),
            # iaa.AddToBrightness((-30, 30))
        ], random_order=True)

        image_paths = os.listdir(input_dir)
        images = read_images(image_paths, input_dir)
        images_aug = seq(images=images)
        save_images(images_aug, image_paths, output_dir, i)

        del images
        del images_aug
        gc.collect()
        print(f"{i} is finished")


def read_images(image_paths, input_dir):
    images = []
    for image_path in image_paths:
       img = imageio.imread(os.path.join(input_dir, image_path))
       images.append(img)
    return images


def save_images(images, image_paths, output_dir, i):
    for u in range(len(images)):
        parts = image_paths[u].split(".")
        imageio.imwrite(os.path.join(output_dir, f"{parts[0]}_aug_{str(i)}.{parts[1]}"), images[u])


if __name__ == '__main__':
    main()