# Copyright contributors to the Terratorch project

import copy
import importlib
from typing import Any
from torchgeo.datamodules.geo import BaseDataModule

def get_class_from_path(class_path: str) -> Any:
    class_parts = class_path.split('.')
    module_name = '.'.join(class_parts[:-1])
    class_path = class_parts[-1]

    # Import the module
    module = importlib.import_module(module_name)

    # Get the class
    cls = getattr(module, class_path)

    return cls

def generate_datamodule(datamodule_args: dict[str: Any]) -> BaseDataModule:

    init_args = datamodule_args["init_args"]
    datamodule_class_path = datamodule_args["class_path"]

    resolved_init_args = copy.deepcopy(init_args)

    if "test_transform" in init_args and init_args["test_transform"]:
        test_transforms = []
        for tt in init_args["test_transform"]:
            tt_class = get_class_from_path(tt["class_path"])
            t_init_args = {} if "init_args" not in tt else tt["init_args"]
            test_transforms.append(tt_class(**t_init_args))

        resolved_init_args["test_transform"] = test_transforms

    if "train_transform" in init_args and init_args["train_transform"]:
        train_transforms = []
        for tt in init_args["train_transform"]:
            tt_class = get_class_from_path(tt["class_path"])
            t_init_args = {} if "init_args" not in tt else tt["init_args"]
            train_transforms.append(tt_class(**t_init_args))

        resolved_init_args["train_transform"] = train_transforms

    if "val_transform" in init_args and init_args["val_transform"]:
        val_transforms = []
        for tt in init_args["val_transform"]:
            tt_class = get_class_from_path(tt["class_path"])
            t_init_args = {} if "init_args" not in tt else tt["init_args"]
            val_transforms.append(tt_class(**t_init_args))

        resolved_init_args["val_transform"] = val_transforms

    print(resolved_init_args)
    datamodule_class = get_class_from_path(datamodule_class_path)
    datamodule = datamodule_class(**resolved_init_args)

    return datamodule
