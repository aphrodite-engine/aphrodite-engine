# Copyright contributors to the Terratorch project

import torch
import torch.nn as nn
from terratorch.registry import MODEL_FACTORY_REGISTRY
from typing import Optional

from pydantic import BaseModel
from typing import List,Dict
from enum import Enum

class InputTypeEnum(str, Enum):
    tensor= 'torch.Tensor'

class data(BaseModel):
    type: InputTypeEnum 
    shape: List[int]

class InputDefinition(BaseModel):
    data: Dict[str,data]
    target: Optional[str] = None

class MultiModalDataGenerator():
    def __init__(self,config: dict):
        self.config = config
        self.input_definition = InputDefinition(**config["input"])
        self.task_type= config['model']["class_path"]
    
    def _get_mm_fields_config(self) -> dict[str:str]:
        fields = {}
        for input_name,input in self.input_definition.data.items():
            if input.type == InputTypeEnum.tensor:
                fields[input_name] = "image"
        return fields

class DummyDataGenerator():
    def __init__(self,config: dict):
        self.config = config
        self.input_definition = InputDefinition(**config["input"])
        self.task_type= config['model']["class_path"]
    
    def get_dummy_mm_data(self) -> dict[str,torch.Tensor]:

        mm_data = {}
        for input_name,input in self.input_definition.data.items():
            if input.type == InputTypeEnum.tensor:
                mm_data[input_name] = torch.full(input.shape,1.0,dtype=torch.float16)
        return mm_data

def lookup_task_name(class_path):
    if "SemanticSegmentationTask" in class_path:
        return 'segmentation'
    elif "PixelwiseRegressionTask" in class_path:
        return 'regression'
    else:
        return None
        
class InferenceRunner(nn.Module):
    
    def __init__(self, config: dict):
        

        super().__init__()
        model_conf=  config["model"]
        self.task_type= config['model']["class_path"]
        task_name = lookup_task_name(model_conf["class_path"])
        model_factory = MODEL_FACTORY_REGISTRY.build(model_conf["init_args"]["model_factory"])
        self.model = model_factory.build_model(task=task_name,**model_conf['init_args']['model_args'])
        self.input_definition = InputDefinition(**config["input"])

    
    def _parse_and_validate_multimodal_data(
        self,**kwargs:object) -> tuple[torch.Tensor, Optional[torch.Tensor]] :
        mm_data = {}
        for input_name,input in self.input_definition.data.items():
            input_value= kwargs.pop(input_name, None)
            if input.type == InputTypeEnum.tensor:
                if not isinstance(input_value, torch.Tensor):
                    raise ValueError(f"Incorrect type of {input_name}. "
                                    f"Got type: {type(input_value)}, expected {input.type}")
                if self.input_definition.target and input_name != self.input_definition.target:
                    input_value= torch.unbind(input_value, dim=0)[0]

            mm_data[input_name] = input_value
        return mm_data

    
    def forward(self,**kwargs: object):
        
        input = self._parse_and_validate_multimodal_data(**kwargs)
        if self.input_definition.target:
            target_input = input.pop(self.input_definition.target)
            model_output = self.model(target_input,**input)
        else:
            model_output = self.model(input)
        return model_output
