import argparse
import pathlib
import time
import jsons
import PIL.Image
import torch
from mitorch.builders import ModelBuilder
from mitorch.common import TrainingConfig
from mitorch.datasets import TransformFactory

COLOR_CODES = ["black", "brown", "red", "orange", "yellow", "green", "blue", "violet", "grey", "white"]


def _draw_od_labels(image, annotations, label_names, output_threshold):
    draw = PIL.ImageDraw.Draw(image)
    for class_id, confidence, x, y, x2, y2 in annotations:
        if confidence > output_threshold:
            x *= image.width
            y *= image.height
            x2 *= image.width
            y2 *= image.height
            color = COLOR_CODES[class_id % len(COLOR_CODES)]
            draw.rectangle(((x, y), (x2, y2)), outline=color)
            draw.text((x, y), f'{label_names[class_id]} ({confidence:.2f})')


def _draw_ic_labels(image, annotations, label_names, output_threshold):
    draw = PIL.ImageDraw.Draw(image)
    for i in range(len(annotations)):
        if annotations[i] > output_threshold:
            draw.text((0, i * 10), f'{label_names[i]} ({annotations[i]})', color='red')


def predict(config_filepath, weights_filepath, num_classes, image_filepaths, output_threshold, output_dir):
    config = jsons.loads(config_filepath.read_text(), TrainingConfig)

    transform = TransformFactory(config.task_type == 'object_detection', config.model.input_size).create(config.augmentation.val)
    model = ModelBuilder(config).build(num_classes, weights_filepath)
    model.eval()

    label_names = [str(i) for i in range(num_classes)]

    for filepath in image_filepaths:
        with PIL.Image.open(filepath) as image:
            transformed, _ = transform(image, [])
            with torch.no_grad():
                start = time.time()
                feature = model(transformed.unsqueeze(0))
                output = model.predictor(feature)[0]
                processing_time = time.time() - start

            if output_dir:
                output_dir.mkdir(exist_ok=True, parents=True)
                output_filepath = output_dir / filepath.name
                drawer = _draw_od_labels if config.task_type == 'object_detection' else _draw_ic_labels
                drawer(image, output, label_names, output_threshold)
                image.save(output_filepath)
                print(f"Saved {filepath} to {output_filepath}. ({processing_time:.3f}s)")
            else:
                print(f"{filepath}: {output} ({processing_time:.3f}s).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_filepath', type=pathlib.Path)
    parser.add_argument('weights_filepath', type=pathlib.Path)
    parser.add_argument('num_classes', type=int)
    parser.add_argument('image_filepath', nargs='*', type=pathlib.Path)
    parser.add_argument('--output_threshold', default=0.5, type=float)
    parser.add_argument('--output_dir', '-o', type=pathlib.Path, help="Draw prediction results on the images and save in this directory.")

    args = parser.parse_args()

    predict(args.config_filepath, args.weights_filepath, args.num_classes, args.image_filepath, args.output_threshold, args.output_dir)


if __name__ == '__main__':
    main()
