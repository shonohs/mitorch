import argparse
import pathlib
import numpy as np
import PIL.Image
import PIL.ImageDraw
from mitorch.datasets import TransformFactory, ImageDataset, ObjectDetectionDataset


COLOR_CODES = ["black", "brown", "red", "orange", "yellow", "green", "blue", "violet", "grey", "white"]


def _draw_od_labels(image, annotations, label_names):
    draw = PIL.ImageDraw.Draw(image)
    w, h = image.width, image.height
    for class_id, x, y, x2, y2 in annotations:
        x, x2 = x * w, x2 * w
        y, y2 = y * h, y2 * h
        color = COLOR_CODES[class_id % len(COLOR_CODES)]
        draw.rectangle(((x, y), (x2, y2)), outline=color)
        draw.text((x, y), label_names[class_id])


def visualize_augmentation(dataset_filepath, output_dir, transform_names, input_size, num_images, num_tries):
    dataset = ImageDataset.from_file(dataset_filepath, lambda x: x)
    is_object_detection = isinstance(dataset, ObjectDetectionDataset)
    transform = TransformFactory(is_object_detection, input_size).create(transform_names)
    if not transform:
        raise RuntimeError(f"Unknown transform: {transform_names}")
    dataset.transform = transform

    for i in range(num_images):
        for j in range(num_tries):
            image, target = dataset[i]
            image = (image + 0.5) * 255
            image = image.permute((1, 2, 0))
            filepath = output_dir / f'{i}_{j}.jpg'
            image = PIL.Image.fromarray(np.array(image, dtype=np.uint8))
            if is_object_detection:
                _draw_od_labels(image, target, dataset.labels)
            image.save(filepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_filepath', type=pathlib.Path)
    parser.add_argument('output_dir', type=pathlib.Path)
    parser.add_argument('transform_name', nargs='+')
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--num_images', '-n', type=int, default=10)
    parser.add_argument('--num_tries', '-t', type=int, default=10)

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    visualize_augmentation(args.dataset_filepath, args.output_dir, args.transform_name, args.input_size, args.num_images, args.num_tries)


if __name__ == '__main__':
    main()
