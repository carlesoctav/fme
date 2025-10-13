from transformers import ModernBertModel as TorchModernBertModel

model_name = "answerdotai/ModernBERT-base"
th_model = TorchModernBertModel.from_pretrained(model_name)

print(f"attention_dropout: {th_model.config.attention_dropout}")
print(f"mlp_dropout: {th_model.config.mlp_dropout}")
print(f"embedding_dropout: {th_model.config.embedding_dropout}")
