import argparse
import os
import tempfile
import uuid
import azureml.core
from azureml.core.authentication import ServicePrincipalAuthentication

from ..environment import Environment

EXPERIMENT_NAME = 'mitorch'


class AzureMLManager:
    """Manage AzureML runs. Submit a new run and query the status of a run.
    This
    """
    def __init__(self, workspace_name, cluster_name, subscription_id, tenant_id=None, username=None, password=None):
        """Initialize the manager. If tenant_id, username and password are given, use service principal authentication."""
        if tenant_id and username and password:
            auth = ServicePrincipalAuthentication(tenant_id=tenant_id, service_principal_id=username, service_principal_password=password)
        else:
            print("Use interactive authentication...")
            auth = None

        self.workspace = azureml.core.Workspace.get(name=workspace_name, subscription_id=subscription_id, auth=auth)
        if not self.workspace:
            raise RuntimeError(f"Workspace {workspace_name} not found")
        self.experiment = azureml.core.Experiment(workspace=self.workspace, name=EXPERIMENT_NAME)
        self.cluster = azureml.core.compute.ComputeTarget(workspace=self.workspace, name=cluster_name)
        if not self.cluster:
            raise RuntimeError(f"Cluster {cluster_name} doesn't exist in workspace {workspace_name}")

    def submit(self, db_uri, job_id):
        run_config = azureml.core.runconfig.RunConfiguration()
        run_config.target = self.cluster
        dependencies = azureml.core.conda_dependencies.CondaDependencies()
        dependencies.set_python_version('3.7')
        run_config.environment.python.conda_dependencies = dependencies
        # Specify a docker base image since the default one is ubuntu 16.04.
        run_config.environment.docker.enabled = True
        run_config.environment.docker.base_image = 'mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04'

        with tempfile.TemporaryDirectory() as temp_dir:
            self._generate_bootstrap(temp_dir)
            args = [str(job_id), db_uri]
            script_run_config = azureml.core.ScriptRunConfig(source_directory=temp_dir, script='boot.py', arguments=args, run_config=run_config)
            run = self.experiment.submit(config=script_run_config)
            run_id = run.get_details()['runId']
            return run_id

    def query(self, run_id):
        """Get the status of the specified run.
        Returns:
            (str) running, failed, or completed.
        """
        run = azureml.core.run.Run(self.experiment, run_id)
        return run.get_status().lower()

    def get_num_available_nodes(self):
        """Get the number of available nodes"""
        status = self.cluster.get_status()
        s = status.serialize()
        return s['scaleSettings']['maxNodeCount'] - s['currentNodeCount']

    def _generate_bootstrap(self, directory):
        filepath = os.path.join(directory, 'boot.py')
        with open(filepath, 'w') as f:
            f.write('import os\n')
            f.write('os.system("pip install https://github.com/shonohs/mitorch/archive/dev.zip")\n')
            f.write('os.system("miamlrun")')


def main():
    parser = argparse.ArgumentParser("Manage MiTorch jobs on AzureML")
    parser.add_argument('command', choices=['submit', 'query'])
    parser.add_argument('--job_id', required=True, help="Training ID to submit. Or AzureML Run ID to query.")

    args = parser.parse_args()
    env = Environment()
    manager = AzureMLManager(env.azureml_workspace_name, env.azureml_cluster_name, env.azureml_subscription_id,
                             env.azureml_tenant_id, env.azureml_username, env.azureml_password)

    if args.command == 'submit':
        manager.submit(env.db_uri, uuid.UUID(args.job_id))
    elif args.command == 'query':
        result = manager.query(args.job_id)
        print(result)


if __name__ == '__main__':
    main()
