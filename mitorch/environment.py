import os


class Environment:
    def __init__(self):
        self._azureml_subscription_id = os.getenv('MITORCH_AZUREML_SUBSCRIPTION_ID')
        self._azureml_workspace = os.getenv('MITORCH_AZUREML_WORKSPACE')
        self._azureml_cluster = os.getenv('MITORCH_AZUREML_CLUSTER')
        # The format of AML_AUTH is <tenant_id>:<service_principal_id>:<password>.
        azureml_auth = os.getenv('MITORCH_AZUREML_AUTH')
        if azureml_auth:
            self._azureml_tenant_id, self._azureml_username, self._azureml_password = azureml_auth.split(':')
        else:
            self._azureml_tenant_id, self._azureml_username, self._azureml_password = None, None, None
        self._db_uri = os.getenv('MITORCH_DB_URI')

    @property
    def azureml_subscription_id(self):
        if not self._azureml_subscription_id:
            raise RuntimeError("MITORCH_AZUREML_SUBSCRIPTION_ID is not set")
        return self._azureml_subscription_id

    @property
    def azureml_workspace_name(self):
        if not self._azureml_workspace:
            raise RuntimeError("MITORCH_AZUREML_WORKSPACE is not set")
        return self._azureml_workspace

    @property
    def azureml_cluster_name(self):
        if not self._azureml_cluster:
            raise RuntimeError("MITORCH_AZUREML_CLUSTER is not set")
        return self._azureml_cluster

    @property
    def azureml_tenant_id(self):
        return self._azureml_tenant_id

    @property
    def azureml_username(self):
        return self._azureml_username

    @property
    def azureml_password(self):
        return self._azureml_password

    @property
    def db_uri(self):
        if not self._db_uri:
            raise RuntimeError("MITORCH_DB_URI is not set")
        return self._db_uri
