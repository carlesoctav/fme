import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import inspect
from transformers.models.modernbert.modeling_modernbert import ModernBertAttention

print("ModernBertAttention.__init__ source:")
print(inspect.getsource(ModernBertAttention.__init__))
