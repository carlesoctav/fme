let's fix our public api


i want this tree thing
import src.training -> expose make_train_step, make_eval_step, train_loop, eval_loop 
Optimizer, init_module, make_module_opt, Eval, make_dataloader (from src.data.training)

src.distributed -> expose fully_shard, tensor_parallel, get_partition_spec, column_paralle, row_parallel, prepare_inpout, prepare_output, prepare_input_output

src.nn -> export all the nn layer/module and functional, (like nn.Linear, etc)

src.models -> export the model that has WeightPlan (like BertModel), and also model namespace like bert
src.callbacks -> export all the callbacks
src.data -> export masked_langauge_modeling, next_token_prediction_transforms


and then we've src -> darray, iter_module, apply_transforms,  
and then src.util -> (gather all util from any submodule for now)


internal code should export the module by relative path (like import ._training)


make a rule for this on AGENTS.md
