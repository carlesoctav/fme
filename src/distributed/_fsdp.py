
def fully_shard(
    module,
    model_axis_name = "data",
    min_model_weights = 2**5,
    manual = False
):
    type_orig_module = type(module)
    FSDP = type(type_orig_module)

