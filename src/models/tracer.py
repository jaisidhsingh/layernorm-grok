import warnings
from typing import *
warnings.simplefilter("ignore")

import torch.nn as nn
from torch import Tensor

from src.models.transformer import Transformer


class Tracer():
    def __init__(self, model: Transformer, hook_points: list):
        self.model = model
        self.hook_points = hook_points
        self.hooks = {}
        self.cache = {}
        self.add_hooks_to_model()
    
    def hook_component(self, name: str): 
        def hook_input_and_output(module, inputs, outputs):
            if isinstance(inputs, tuple):
                inp = inputs[0]
            else:
                inp = inputs
            if isinstance(outputs, tuple):
                out = outputs[0]
            else:
                out = outputs
                
            b, l, d = inp.shape
            assert out.shape == inp.shape, out.shape
            try:
                tokenwise_inputs = inp.clone().view(b*l, d)
            except: 
                tokenwise_inputs = inp.clone().reshape(b*l, d)
            try:
                tokenwise_outputs = out.clone().view(b*l, d)
            except:
                tokenwise_outputs = out.clone().reshape(b*l, d)
            self.cache[name] = {"input": tokenwise_inputs, "output": tokenwise_outputs}
        
        return hook_input_and_output
    
    def add_hooks_to_model(self): 
        for i, layer in enumerate(self.model.layers):
            name = f"layer_{i}" 
            self.hooks[name] = layer.register_forward_hook(self.hook_component(name))
            
            for point in self.hook_points:
                sub_name = f"{name}_{point}"
                sublayer: nn.Module = getattr(layer, point)
                self.hooks[sub_name] = sublayer.register_forward_hook(self.hook_component(sub_name))
        
        if hasattr(self.model, "final_ln"):
            self.hooks["final_ln"] = self.model.final_ln.register_forward_hook(self.hook_component("final_ln"))
    
    def __call__(self, x: Tensor):
        return self.model(x)              
    
    def clear_cache(self):
        self.cache = {}
    
    def remove_hooks(self):
        for v in self.hooks.values():
            v.remove()
