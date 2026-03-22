from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .config_msgla import MSGLAConfig
from .modeling_msgla import MSGLAForCausalLM, MSGLAModel

__all__ = ["MSGLAConfig", "MSGLAForCausalLM", "MSGLAModel"]

AutoConfig.register(MSGLAConfig.model_type, MSGLAConfig)
AutoModel.register(MSGLAConfig, MSGLAModel)
AutoModelForCausalLM.register(MSGLAConfig, MSGLAForCausalLM)
