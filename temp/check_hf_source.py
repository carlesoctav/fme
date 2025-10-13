import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from transformers import ModernBertModel as TorchModernBertModel
import inspect

model_name = "answerdotai/ModernBERT-base"
th_model = TorchModernBertModel.from_pretrained(model_name)

# Get the encoder layer __init__ code
from transformers.models.modernbert.modeling_modernbert import ModernBertEncoderLayer
print("ModernBertEncoderLayer.__init__ source:")
print(inspect.getsource(ModernBertEncoderLayer.__init__))
