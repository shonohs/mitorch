import random
from .task import Task


class RandomElement:
    pass


class RandomChoice(RandomElement):
    def __init__(self, choice_list):
        assert all(isinstance(c, (int, float, str)) for c in choice_list), f"Invalid choices: {choice_list}"
        self.choice_list = choice_list
        self.current_choice = None

    def update(self):
        self.current_choice = random.choice(self.choice_list)

    def get(self):
        return self.current_choice


def _create_element(data):
    if '_choice' in data:
        return RandomChoice(data['_choice'])
    return None


class RandomSearchTask(Task):
    def __init__(self, task_description):
        super().__init__(task_description)
        self.config = task_description['config']
        self._elements = []
        self._parsed_config = self._parse_config(self.config)

    def _parse_config(self, config):
        if isinstance(config, list):
            return [self._parse_config(c) for c in config]
        elif isinstance(config, dict):
            element = _create_element(config)
            if element:
                self._elements.append(element)
                return element

            return {key: self._parse_config(config[key]) for key in config}
        else:
            return config

    def _get_config(self, config):
        if isinstance(config, RandomElement):
            return config.get()
        if isinstance(config, list):
            return [self._get_config(c) for c in config]
        elif isinstance(config, dict):
            return {key: self._get_config(config[key]) for key in config}
        else:
            return config

    def fetch_next(self):
        if not self.has_next():
            return None

        random.seed(0)
        for i in range(self.num_trainings + 1):
            self._update()  # Fast forward to the current index.

        return self._get_config(self._parsed_config)

    def _update(self):
        for element in self._elements:
            element.update()
