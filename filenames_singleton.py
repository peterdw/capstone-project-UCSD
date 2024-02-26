import glob
import pathlib

from constants import MAESTRO_DATASET_FOLDER


class FilenamesSingleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(FilenamesSingleton, cls).__new__(cls)
            cls._initialize(*args, **kwargs)
        return cls._instance

    @classmethod
    def _initialize(cls, *args, **kwargs):
        # Assuming data_dir is passed as an argument or set here directly
        data_dir = kwargs.get('data_dir', pathlib.Path(MAESTRO_DATASET_FOLDER))
        cls.filenames = glob.glob(str(data_dir / '**/*.mid*'))
