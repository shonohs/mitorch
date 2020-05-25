import dataclasses
from typing import Dict, List


@dataclasses.dataclass
class AzureMLSetting:
    region_name: str  # Region name for this resource. Must be unique.
    subscription_id: str
    workspace_name: str
    cluster_name: str
    sp_tenant_id: str = None  # Service Principal tenant id (optional)
    sp_username: str = None  # Service Principal username (optional)
    sp_password: str = None  # Service Principal password (optional)


@dataclasses.dataclass
class Settings:
    # Azure Blob url with SAS token to store trained models.
    storage_url: str

    # Dictionary of dataset for each regions.
    dataset_url: Dict[str, str]

    # AzureML settings
    azureml_settings: List[AzureMLSetting]

    @classmethod
    def from_dict(cls, data):
        azureml_settings = [AzureMLSetting(**s) for s in data['azureml_settings']]
        return cls(storage_url=data['storage_url'], dataset_url=data['dataset_url'], azureml_settings=azureml_settings)
