

03 10 25
attention masking, attention function refactor
(use flex attention api design)


06 10 25
load tokenized dataset, need to infer a segments and make position_ids
refactor readoptions v
refactor dataloader v
fix modernbert hardbro im giving up
fix modelcheckpoint v
need to add load_checkpoint

model compile took a very-very long time i rng there.

07 10 25

model checkpoint fix
largescale distirbuted tokenizing with raycluster
attention dropoout mask fix


ckpt = 
abstract_module = eqx.filter_eval_shape(module, etc, etc)  -> jax.shapedstruct
abstract_param = filter(abstarct_moudle, eqx.is_array) -> this probaly will fail no?

ckpt.restore(step, v)

