class StepMetric():
    
a= LossMetric(StepMetric):

class Stepmetric:
    loss: (jax.array, count, mode)
    acc: (jax.array, count, mode)

def train_step():
    pass


a.update(loss = loss, )

def loss_fn(model, batch, *, key):

    metric= {}

    aux = {
        "loss": { vlaue: jax.array, count:  
    }

    return loss, aux


fabric = trainerUtil(trainingconfig)

model, optimizer = fabric.setup(model, optimizer)

dataloader = fabric(datasets) put to the correct sharding



def fabric.make_train_step(
    loss_fn,
    *loss_fn_args,
    **loss_fn_kwargs,
    debug = True,

):
    train_step(*loss_fn_args, **loss_fn_kwargs):
        grad_fn = eqx.filter_value_and_grad(loss_fn)
        grad, (_, aux) = grad_fn(loss_fn_args, loss_fn_kwargs)
        new_state, optimizer = optimizer(grad, model)
        
    if not debug:
        return jax.jit(train_step)
    else:
        train_step

for batch in data_laoder:
    model, optimzer, (metrics) = train_step(model, batch, *, key = key) if there's state
    model, optimzer, loss, aux = train_step(model, batch, *, key = key) if there's state
    fabric.log(loss, aux)

---------


train_step():
    


with trainer_util.log():
    


