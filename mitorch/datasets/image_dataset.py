import os
import pathlib
import zipfile
import PIL.Image


class ImageDataset:
    def __init__(self, filename, transform):
        filepath = pathlib.Path(filename)
        self.transform = transform
        self.images = []
        self.base_dir = filepath.parent
        self.reader = FileReader(self.base_dir)

        with open(filename) as f:
            for line in f:
                image_filepath, target = line.strip().split()
                image_filepath = image_filepath.strip()
                target = self._load_target(target)
                self.images.append((image_filepath, target))

        max_label = self._get_max_label()
        self._labels = self._load_labels(max_label)

    def _load_labels(self, max_label):
        """Load if there is labels.txt. If not, generate dummy labels"""
        labels_filepath = self.base_dir / 'labels.txt'
        if labels_filepath.exists():
            with open(labels_filepath) as f:
                labels = [l.strip() for l in f.readlines()]
                assert len(labels) > max_label
                return labels
        else:
            return [f'label_{i}' for i in range(max_label + 1)]

    @property
    def labels(self):
        return self._labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_filepath, target = self.images[index]
        with self.reader.open(image_filepath, 'rb') as f:
            image = PIL.Image.open(f)
            image.load()
        return self.transform(image, target)

    def _load_target(self, target):
        raise NotImplementedError

    def _get_max_label(self):
        raise NotImplementedError

    @classmethod
    def from_file(cls, filename, transform):
        dataset_type = cls._detect_type(filename)
        if dataset_type == 'multiclass_classification':
            return MulticlassClassificationDataset(filename, transform)
        elif dataset_type == 'multilabel_classification':
            return MultilabelClassificationDataset(filename, transform)
        elif dataset_type == 'object_detection':
            return ObjectDetectionDataset(filename, transform)
        else:
            raise NotImplementedError(f"Non supported dataset type: {dataset_type}")

    @staticmethod
    def _detect_type(filename):
        with open(filename) as f:
            for line in f:
                _, labels = line.split()
                if '@' in labels or '.' in labels:
                    return 'object_detection'
                if ',' in labels:
                    return 'multilabel_classification'
        return 'multiclass_classification'


class ThreadSafeZipFile(zipfile.ZipFile):
    def __init__(self, zip_filepath):
        self.zip_filepath = zip_filepath
        self.cache = {}

    def open(self, filepath):
        if os.getpid() not in self.cache:
            self.cache[os.getpid()] = zipfile.ZipFile(self.zip_filepath)
        return self.cache[os.getpid()].open(filepath, 'r')

    def close(self):
        for zip_file in self.cache.values():
            zip_file.close()


class FileReader:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.zipfile_cache = {}

    def open(self, filepath, mode='r'):
        if '@' in filepath:
            zip_filepath, filepath = filepath.split('@')
            if zip_filepath not in self.zipfile_cache:
                self.zipfile_cache[zip_filepath] = ThreadSafeZipFile(self.base_dir / zip_filepath)
            return self.zipfile_cache[zip_filepath].open(filepath)
        else:
            return open(self.base_dir / filepath, mode)

    def __getstate__(self):
        return {'base_dir': self.base_dir}

    def __setstate__(self, state):
        self.base_dir = state['base_dir']
        self.zipfile_cache = {}


class MulticlassClassificationDataset(ImageDataset):
    def _get_max_label(self):
        return max(i[1] for i in self.images)

    @staticmethod
    def _load_target(target):
        return int(target)


class MultilabelClassificationDataset(ImageDataset):
    def _get_max_label(self):
        return max(j for i in self.images for j in i[1])

    @staticmethod
    def _load_target(target):
        return [int(t) for t in target.split(',')]


class ObjectDetectionDataset(ImageDataset):
    def _get_max_label(self):
        return max(j[0] for i in self.images for j in i[1])

    def _load_target(self, targetpath):
        with self.reader.open(targetpath) as f:
            # label, x_min, y_min, x_max, y_max. Those are not normalized.
            return [(int(t[0]), float(t[1]), float(t[2]), float(t[3]), float(t[4])) for line in f for t in line.strip().split()]
