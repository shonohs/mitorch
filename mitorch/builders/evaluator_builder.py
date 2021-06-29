from mitorch.evaluators import MulticlassClassificationEvaluator, MultilabelClassificationEvaluator, ObjectDetectionEvaluator


class EvaluatorBuilder:
    def __init__(self, config):
        self._task_type = config.task_type

    def build(self):
        mappings = {'multiclass_classification': MulticlassClassificationEvaluator,
                    'multilabel_classification': MultilabelClassificationEvaluator,
                    'object_detection': ObjectDetectionEvaluator}
        assert self._task_type in mappings
        return mappings[self._task_type]()
