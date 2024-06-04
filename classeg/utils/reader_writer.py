import pickle
from typing import Type, Union

import SimpleITK as sitk
import numpy as np
import torch

from classeg.utils.constants import *
from PIL import Image
from overrides import override


class BaseReaderWriter:
    def __init__(self, case_name: str, dataset_name: str = None):
        super().__init__()
        self.direction = None
        self.spacing = None
        self.origin = None
        self.has_read = False
        self.case_name = case_name
        self.dataset_name = dataset_name

    def __verify_extension(self, extension: str) -> None:
        raise NotImplementedError("Do not use BaseReader, but instead use a subclass that overrides read.")

    def read(self, path: str, **kwargs) -> np.array:
        raise NotImplementedError("Do not use BaseReader, but instead use a subclass that overrides read.")

    def write(self, data: Union[Type[np.array], Type[torch.Tensor]], path: str, standardize=False) -> None:
        raise NotImplementedError("Do not use BaseWriter, but instead use a subclass that overrides write.")

    def standardize(self, data: Union[Type[np.array], Type[torch.Tensor]]) -> Union[Type[np.array], Type[torch.Tensor]]:
        raise NotImplementedError("Do not use BaseReader, but instead use a subclass that overrides standardize.")

    def __store_metadata(self) -> None: ...

    @property
    def image_dimensions(self) -> int:
        """
        Number of SPACIAL dimensions that this read/writer manages. Ignores channels.
        """
        raise NotImplementedError("Do not use BaseWriter, but instead use a subclass that overrides write.")


class SimpleITKReaderWriter(BaseReaderWriter):

    def __verify_extension(self, extension: str) -> None:
        assert extension == 'nii.gz', f'Invalid extension {extension} for reader SimpleITKReader.'

    def __store_metadata(self) -> None:
        assert self.dataset_name is not None, "Can not store metadata from SimpleITK reader/writer without knowing " \
                                              "dataset name."
        expected_folder = f"{PREPROCESSED_ROOT}/{self.dataset_name}/metadata"
        expected_file = f"{expected_folder}/{self.case_name}.pkl"
        # Assumption being made here is that images and masks will have the same metadata within a case.
        data = {
            'spacing': self.spacing,
            'direction': self.direction,
            'origin': self.origin
        }
        if os.path.exists(expected_file):
            os.remove(expected_file)
        if not os.path.exists(expected_folder):
            try:
                os.makedirs(expected_folder)
            except FileExistsError:
                pass
        with open(expected_file, 'wb') as file:
            return pickle.dump(data, file)

    def read(self, path: str, store_metadata: bool = False, **kwargs) -> np.array:
        self.has_read = True
        self.__verify_extension('.'.join(path.split('.')[1:]))
        image = sitk.ReadImage(path)
        self.spacing = image.GetSpacing()
        self.direction = image.GetDirection()
        self.origin = image.GetOrigin()
        if store_metadata:
            self.__store_metadata()
        return sitk.GetArrayFromImage(image)

    def standardize(self, data: Union[Type[np.array], Type[torch.Tensor]]) -> Union[Type[np.array], Type[torch.Tensor]]:
        if len(data.shape) == 3:
            return data[np.newaxis, ...]
        elif len(data.shape) == 4:
            if data.shape[-1] in [1]:
                data = np.transpose(data, (3, 0, 1, 2))
            return data
        else:
            raise ValueError(f"Invalid shape {data.shape} for standardization in SimpleITKReaderWriter.")

    def check_for_metadata_folder(self) -> Union[dict, None]:
        expected_folder = f"{PREPROCESSED_ROOT}/{self.dataset_name}/metadata"
        expected_file = f"{expected_folder}/{self.case_name}.pkl"
        if os.path.exists(expected_file):
            with open(expected_file, 'rb') as file:
                return pickle.load(file)
        return None

    def write(self, data: Union[Type[np.array], Type[torch.Tensor]], path: str, standardize=False) -> None:
        if standardize:
            data = self.standardize(data)
        if not self.has_read:
            meta = self.check_for_metadata_folder()
            if meta is None:
                raise ValueError(f'SimpleITK reader writer can not find metadata for this image {self.case_name}. If '
                                 f'you read first we can save.')
            try:
                self.spacing = meta['spacing']
                self.direction = meta['direction']
                self.origin = meta['origin']
            except KeyError:
                raise ValueError(f'Invalid metadata found for {self.case_name} in SimpleITKReaderWriter.')
        self.__verify_extension('.'.join(path.split('.')[1:]))
        if isinstance(data, torch.Tensor):
            data = np.array(data.detach().cpu())
        image = sitk.GetImageFromArray(data)
        image.SetSpacing(self.spacing)
        image.SetOrigin(self.origin)
        image.SetDirection(self.direction)
        sitk.WriteImage(image, path)

    @property
    def image_dimensions(self) -> int:
        return 3


class NaturalReaderWriter(BaseReaderWriter):

    def __verify_extension(self, extension: str) -> None:
        assert extension in ['png', 'jpg', 'npy', 'jpeg', 'JPEG'], (f'Invalid extension {extension} for reader '
                                                                    f'NaturalReaderWriter.')

    def read(self, path: str, store_metadata: bool = False, **kwargs) -> np.array:
        name = path.split('/')[-1]
        extension = '.'.join(name.split('.')[1:])
        self.__verify_extension(extension)
        if extension == 'npy':
            image = np.load(path, allow_pickle=True)
        else:
            image = np.array(Image.open(path))
        return image

    def standardize(self, data: Union[Type[np.array], Type[torch.Tensor]]) -> Union[Type[np.array], Type[torch.Tensor]]:
        """
        Standardizes the data to the format expected by the pipeline.

        Channel first
        """
        if len(data.shape) == 2:
            return data[np.newaxis, ...]
        elif len(data.shape) == 3:
            if data.shape[-1] in [1, 3]:
                return np.transpose(data, (2, 0, 1))
            else:
                return data
        else:
            raise ValueError(f"Invalid shape {data.shape} for standardization in NaturalImages.")

    def write(self, data: Union[Type[np.array], Type[torch.Tensor]], path: str, standardize=False) -> None:
        if standardize:
            data = self.standardize(data)
        name = path.split('/')[-1]
        self.__verify_extension('.'.join(name.split('.')[1:]))
        if isinstance(data, torch.Tensor):
            data = np.array(data.detach().cpu())
        np.save(path, data)

    @property
    def image_dimensions(self) -> int:
        return 2


def get_reader_writer(io: str) -> Type[BaseReaderWriter]:
    assert io in [SIMPLE_ITK], f'Unrecognized reader/writer {io}.'
    reader_writer_mapping = {
        SIMPLE_ITK: SimpleITKReaderWriter,
        NATURAL: NaturalReaderWriter
    }
    return reader_writer_mapping[io]


def get_reader_writer_from_extension(extension: str) -> Type[BaseReaderWriter]:
    # TODO better imagenet detection for reader writer
    mapping = {
        'nii.gz': SimpleITKReaderWriter,
        'png': NaturalReaderWriter,
        'jpg': NaturalReaderWriter,
        'jpeg': NaturalReaderWriter,
        'npy': NaturalReaderWriter
    }
    assert extension in mapping.keys(), f"Currently unsupported extension {extension}"
    return mapping[extension]
