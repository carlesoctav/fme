

class StepMetric():
    
a= LossMetric(StepMetric):

class Stepmetric:
    loss: (jax.array, count, mode)
    acc: (jax.array, count, mode)




a.update(loss = loss, )

def loss_fn(model, batch, *, key):

    metric= {}
    metric = {
        "loss": { vlaue: jax.array, count:  
    }

    return loss, aux


fabric = trainerUtil(trainingconfig)

model, optimizer = fabric.setup(model, optimizer)

dataloader = fabric(datasets) put to the correct sharding



def fabric.make_train_step(loss_fn, *loss_fn_args, **loss_fn_kwargs, has_aux = True, debug = True, has_state = False):
    train_step(*loss_fn_args, **loss_fn_kwargs):
        grad_fn = eqx.filter_value_and_grad(loss_fn)
        grad, (loss, aux) = grad_fn(loss_fn_args, loss_fn_kwargs)
    if not debug:
        return jax.jit(train_step)
    else:
        train_step

train_step = fabric.make_train_step()



for batch in data_laoder:
    model, optimzer, loss_fn_output (loss, aux, etc) = train_step(model, batch, *, key = key) if there's state
    model, optimzer, loss, aux = train_step(model, batch, *, key = key) if there's state
    fabric.log(loss, aux)




