from collections.abc import Callable

import equinox as eqx
import grain
import jax
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
from transformers.models.bert.configuration_bert import BertConfig

from src import Optimizer
from src.data import DataTransformsForMaskedLMGivenText
from src.distributed import get_partition_spec
from src.losses import softmax_cross_entropy_with_integer_labels
from src.models.bert.modeling_bert import BertForMaskedLM, BertModel


def masked_lm_loss_function(model: BertForMaskedLM, batch, ignore_label: int = -100):
    """Compute masked LM loss, ignoring positions with label == ignore_index.

    - Vectorizes the model over batch.
    - Uses safe labels for ignored positions to avoid invalid indexing.
    - Normalizes by the number of unmasked tokens.
    """
    x = batch["input"]
    y = batch["labels"]
    logits = model(**x) 
    per_token_loss = softmax_cross_entropy_with_integer_labels(logits, y, reduction = "mean")

    return per_token_loss, None

def init_data_loader(
    columns_to_tokenize: str,
    ds_name: str,
    num_epochs: int,
    tokenizer: PreTrainedTokenizerBase,
):
    ds = load_dataset(ds_name, split = "QED")
    ds = grain.MapDataset.source(ds)
    transformations = [DataTransformsForMaskedLMGivenText(tokenizer, columns = "text", max_length = 512)]
    data_loader = grain.DataLoader(
        data_source = ds,
        sampler = grain.samplers.IndexSampler(len(ds), shuffle = True, num_epochs = num_epochs),
        operations = transformations
    )
    
    return data_loader


def init_model(
    model_config,
    optimizer_fn,
    mesh = None,
    shard = True
):
    model_config = BertConfig()
    if not shard:
        model = BertModel(config = model_config)
        optimizer = Optimizer(model, optimizer_fn)
        return model, optimizer
    
    @jax.jit
    def init_jit_model(config):
        model =  BertModel(config = model_config)
        dp_partition_spec = get_partition_spec(model)
        model= eqx.filter_shard(model, dp_partition_spec)
        optimizer = Optimizer(optimizer_fn, model)
        return model, optimizer


    model, optimizer = init_jit_model(model_config)
    return model, optimizer


def init_train_step(loss_fn: Callable):
    def train_step(
        model, 
        optimizer,
        batch, 
    ):
        grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux = True)

        (loss, aux), grad = grad_fn(model, batch)
        model, optimizer = optimizer(grad, model)

        metrics = {"loss_step": loss, "aux_step": aux}

        return model, optimizer, metrics

    return eqx.filter_jit(train_step)


def maybe_checkpoint():
    pass


def maybe_log_metrics():
    pass


def main(epochs = 10):
    model, optimizer = init_model(debug = False)
    data_loader = init_data_loader()
    train_step = init_train_step(masked_lm_loss_function)

    for epoch in epochs:
        for local_step, batch in enumerate(data_loader):
            model, optimizer, metrics = train_step(model, optimizer, batch)

            maybe_log_metrics(metrics, local_step)
            maybe_checkpoint(model, optimizer, data_loader, local_step)
        
        maybe_log_metrics(metrics, epoch)
        maybe_checkpoint(model, optimizer, data_loader, epoch)
        
if __name__ == "__main__":
    main()
