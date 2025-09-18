from src.models.bert import BertModel
from transformers import BertConfig
import jax
import transformers
import equinox as eqx
import numpy as np

tokenizer: transformers.BertTokenizer = transformers.AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

jx_model = BertModel.from_huggingface("google-bert/bert-base-uncased", key = jax.random.key(10))
jx_model = eqx.nn.inference_mode(jx_model)

model = transformers.BertModel.from_pretrained("google-bert/bert-base-uncased")


token = tokenizer("hallo, saya makan nasi goreng", return_tensors = "pt", )
token_but_np = tokenizer("hallo, saya makan nasi goreng", return_tensors = "np")
print(f"DEBUGPRINT[313]: test_hf_output.py:13: token_but_np={token_but_np}")



output_np = jx_model(**token_but_np)
output = model(**token).last_hidden_state.detach().cpu().numpy()

print(f"DEBUGPRINT[311]: test_hf_output.py:16: output_np={output_np}")
print(f"DEBUGPRINT[312]: test_hf_output.py:18: output={output}")


assert np.allclose(output_np, output, atol = 1e-3, rtol = 1e-3)
