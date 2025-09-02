let's create the TrainerModule similar to xlstm trainermodule,

train_module = TrainerModule(kafl;jslfk)
trainer_module.init()


module, optimizer = setup(module: eqx.module | type[eqx.module], ) -> module but sharded
module, state , optimizer = setup(module: eqx.module | type[eqx.module], is_stateful) -> module but sharded
if module is stateful

(this thing is similar like init_mdel, init_optimizer)

loss_fn(module, *args, ):
    return loss, (metrics, mutable_state/None)

or

loss_fn(module, *args,):
    return loss, aux, 
    

where aux == metrics

make_train_step(loss_fn, model, optimizer)
    under the hood it built this
    def train_step(model, *args, opt_state: states | None, )
        grad_fn = eqx.filter_value_and_grad(loss_fn, model, has_aux = True)
        grad, loss, aux  = grad_fn(*args)
        model, opt_state = optimizer.updates(grad, model)
        return model, opt_state
