from vllm import ModelRegistry
from .modeling import FuxiTranyuForCausalLM

def register():
    ModelRegistry.register_model("FuxiTranyuForCausalLM", FuxiTranyuForCausalLM)