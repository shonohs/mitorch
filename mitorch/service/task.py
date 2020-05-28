import copy


class Task:
    def __init__(self, task_description):
        self._task_description = task_description
        self.state = task_description.get('state', {})
        self.status = task_description.get('status', 'new')
        self.num_trainings = task_description.get('num_trainings', 0)
        self.max_trainings = task_description['max_trainings']

    @staticmethod
    def from_dict(data):
        from .random_search_task import RandomSearchTask
        name = data['name']
        task_class = {'random_search': RandomSearchTask}[name]
        return task_class(data)

    def fetch_next(self):
        raise NotImplementedError

    def update_training_status(self, training):
        if training['status'] in ['completed', 'failed']:
            self.num_trainings += 1
            if self.num_trainings >= self.max_trainings:
                self.status = 'completed'

    def has_next(self):
        return self.status == 'active' and self.num_trainings < self.max_trainings

    def to_dict(self):
        task = copy.deepcopy(self._task_description)
        task['state'] = self.state
        task['status'] = self.status
        task['num_trainings'] = self.num_trainings
        return task

    def __str__(self):
        return str(self.to_dict())
