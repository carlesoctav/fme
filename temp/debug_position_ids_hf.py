import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import torch
from transformers import AutoTokenizer, ModernBertModel as TorchModernBertModel

model_name = "answerdotai/ModernBERT-base"
text1 = "hallo nama saya carles dan ini adalah text yang lebih panjang"
text2 = "hallo dunia"

tokenizer = AutoTokenizer.from_pretrained(model_name)
th_model = TorchModernBertModel.from_pretrained(model_name)
th_model.eval()

encoded = tokenizer([text1, text2], padding=True, return_tensors="pt")
input_ids = encoded["input_ids"]
attention_mask = encoded["attention_mask"]

print("Input IDs shape:", input_ids.shape)
print("Attention mask:")
print(attention_mask)
print()

# Monkey patch to capture position_ids
original_forward = th_model.forward
captured_position_ids = None

def patched_forward(*args, **kwargs):
    global captured_position_ids
    # HF creates position_ids inside forward if not provided
    # Let's explicitly pass None to see what it creates
    
    # Check the actual forward implementation
    import inspect
    print("Forward called with kwargs:", list(kwargs.keys()))
    
    # Create position_ids the way HF does it
    input_ids_arg = kwargs.get('input_ids', args[0] if args else None)
    attention_mask_arg = kwargs.get('attention_mask', None)
    
    if input_ids_arg is not None:
        batch_size, seq_length = input_ids_arg.shape
        
        # This is how HuggingFace creates position_ids
        if attention_mask_arg is not None:
            # Create position_ids based on attention_mask
            position_ids = attention_mask_arg.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask_arg == 0, 0)
        else:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids_arg.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        print("\nCreated position_ids:")
        print(position_ids)
        captured_position_ids = position_ids
    
    return original_forward(*args, **kwargs)

th_model.forward = patched_forward

with torch.no_grad():
    output = th_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

print("\nOutput shape:", output.last_hidden_state.shape)
