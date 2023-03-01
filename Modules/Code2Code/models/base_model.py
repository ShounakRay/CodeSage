from typing import Any
from huggingface_hub import login
class BaseCode2CodeModel:
    
    def __init__(self):
        self.prefix = "refine: "
        self.metric = None
        login()
    
    def postprocess_function(self, preds, labels):
        raise NotImplementedError
    
    def preprocess_function(self, example):
        raise NotImplementedError
    
    def compute_metrics(eval_preds):
        raise NotImplementedError
    
    def train(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
        
    def __call__():
        raise NotImplementedError