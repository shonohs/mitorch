import os
import tempfile
import azureml.core
from azureml.core.authentication import ServicePrincipalAuthentication
from ..settings import AzureMLSetting

EXPERIMENT_NAME = 'mitorch'


class AzureMLManager:
    """Manage AzureML runs. Submit a new run and query the status of a run.
    """
    def __init__(self, settings):
        assert isinstance(settings, list)
        self.managers = [AzureMLSingleResourceManager(s) for s in settings]

    def get_num_available_nodes(self):
        return sum(m.get_num_available_nodes() for m in self.managers)

    def submit(self, *args):
        for manager in self.managers:
            if manager.get_num_available_nodes() > 0:
                aml_run_id = manager.submit(*args)
                return aml_run_id, manager.region
        return None

    def query(self, run_id, region):
        for manager in self.managers:
            if manager.region == region:
                return manager.query(run_id)
        return None


class AzureMLSingleResourceManager:
    def __init__(self, setting):
        assert isinstance(setting, AzureMLSetting)
        self.setting = setting
        if setting.sp_tenant_id and setting.sp_username and setting.sp_password:
            auth = ServicePrincipalAuthentication(tenant_id=setting.sp_tenant_id,
                                                  service_principal_id=setting.sp_username,
                                                  service_principal_password=setting.sp_password)
        else:
            print("Use interactive authentication...")
            auth = None

        self.workspace = azureml.core.Workspace.get(name=setting.workspace_name, subscription_id=setting.subscription_id, auth=auth)
        if not self.workspace:
            raise RuntimeError(f"Workspace {setting.workspace_name} not found")
        self.experiment = azureml.core.Experiment(workspace=self.workspace, name=EXPERIMENT_NAME)
        self.cluster = azureml.core.compute.ComputeTarget(workspace=self.workspace, name=setting.cluster_name)
        if not self.cluster:
            raise RuntimeError(f"Cluster {setting.cluster_name} doesn't exist in workspace {setting.workspace_name}")

    @property
    def region(self):
        return self.setting.region_name

    def submit(self, db_url, job_id):
        run_config = azureml.core.runconfig.RunConfiguration()
        run_config.target = self.cluster
        dependencies = azureml.core.conda_dependencies.CondaDependencies()
        dependencies.set_python_version('3.7')
        dependencies.add_pip_package('torch==1.5.1')
        run_config.environment.python.conda_dependencies = dependencies
        # Specify a docker base image since the default one is ubuntu 16.04.
        run_config.environment.docker.enabled = True
        run_config.environment.docker.base_image = 'mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04'
        run_config.environment.docker.shm_size = '16g'

        with tempfile.TemporaryDirectory() as temp_dir:
            self._generate_bootstrap(temp_dir)
            args = [str(job_id), '"' + db_url + '"']
            script_run_config = azureml.core.ScriptRunConfig(source_directory=temp_dir, script='boot.py', arguments=args, run_config=run_config)
            run = self.experiment.submit(config=script_run_config)
            run_id = run.get_details()['runId']
            return run_id

    def query(self, run_id):
        """Get the status of the specified run.
        Returns:
            (str) running, failed, or completed.
        """
        try:
            run = azureml.core.run.Run(self.experiment, run_id)
            return run.get_status().lower()
        except Exception as e:
            print(e)
            return None

    def get_num_available_nodes(self):
        """Get the number of available nodes"""
        status = self.cluster.get_status()
        s = status.serialize()
        return s['scaleSettings']['maxNodeCount'] - s['currentNodeCount']

    @staticmethod
    def _generate_bootstrap(directory):
        filepath = os.path.join(directory, 'boot.py')
        with open(filepath, 'w') as f:
            f.write('import os\n')
            f.write('import sys\n')
            f.write('os.system("pip install https://github.com/shonohs/mitorch_models/archive/dev.zip")\n')
            f.write('os.system("pip install https://github.com/shonohs/mitorch/archive/dev.zip")\n')
            f.write('os.system("miamlrun " + " ".join(sys.argv[1:]))')
