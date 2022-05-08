from hydra.core.config_store import ConfigStore
from .conf import Config
from .data_conf import DataSplitParams, DatasetParams
from .model_conf import GaussNBParams

__all__ = ['Config',
           'DataSplitParams',
           'DatasetParams']

cs = ConfigStore.instance()
cs.store(name='config', node=Config)
cs.store(name='split', node=DataSplitParams)
cs.store(name='dataset', node=DatasetParams)
cs.store(group='model', name='gaussnb', node=GaussNBParams)
