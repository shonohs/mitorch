import dataclasses
from typing import Dict, List


@dataclasses.dataclass
class AzureMLSetting:
    subscription_id: str
    workspace_name: str
    cluster_name: str
    sp_tenant_id: str  # Service Principal tenant id (optional)
    sp_username: str  # Service Principal username (optional)
    sp_password: str  # Service Principal password (optional)
    region_name: str  # Region name for this resource. Must be unique.


@dataclasses.dataclass
class Settings:
    # Azure Blob url with SAS token to store trained models.
    storage_url: str

    # Dictionary of dataset for each regions.
    dataset_url: Dict[str, str]

    # AzureML settings
    azureml_settings: List[AzureMLSetting]
