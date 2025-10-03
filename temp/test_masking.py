from src._masking_utils import create_causal_mask, MaskRequest, EagerAttentionMaskImplementation


req = MaskRequest(1, 1, 10, 10)

mask_impl = EagerAttentionMaskImplementation()



attn_mask = create_causal_mask(
    mask_impl = mask_impl, 
    request = req
)

print(f"DEBUGPRINT[355]: test_masking.py:10: attn_mask={attn_mask}")
