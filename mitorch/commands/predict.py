import argparse
import pathlib
import time
import jsons
import PIL.Image
import torch
from mitorch.builders import ModelBuilder
from mitorch.common import TrainingConfig
from mitorch.datasets import TransformFactory


def predict(config_filepath, weights_filepath, num_classes, image_filepaths):
    config = jsons.loads(config_filepath.read_text(), TrainingConfig)

    transform = TransformFactory(config.task_type == 'object_detection', config.model.input_size).create(config.augmentation.val)
    model = ModelBuilder(config).build(num_classes, weights_filepath)
    model.eval()

    for filepath in image_filepaths:
        with PIL.Image.open(filepath) as image:
            transformed, _ = transform(image, [])
            with torch.no_grad():
                start = time.time()
                feature = model(transformed.unsqueeze(0))
                output = model.predictor(feature)
                print(f"Processed in {time.time() - start}s.")
            print(f"{filepath}: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_filepath', type=pathlib.Path)
    parser.add_argument('weights_filepath', type=pathlib.Path)
    parser.add_argument('num_classes', type=int)
    parser.add_argument('image_filepath', nargs='*', type=pathlib.Path)

    args = parser.parse_args()

    predict(args.config_filepath, args.weights_filepath, args.num_classes, args.image_filepath)


if __name__ == '__main__':
    main()
