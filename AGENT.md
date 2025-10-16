let's fix our codebase

first

train_loop(
    module, 
    optimizer,
    train_step_fn:
    train_loader,
    logger, 
    train_metrics
    num_train_step
    enable_wallclock
)


on callback now it's rquired for use to add every: int
    now instead of doing the check in the callback, w'll do check this on train_loop

    for callback in callbacks:
        if step % callback.every == 0:
            callback(module, optmizer, batch, logs, logger) so remove the step

also because of modelcheckpoint is has own internal step, we set very = 1 for modelchckpoint


==========

next let's talk about sharding, on tp.py

instad of creating new module and new calss every time we do a transformation please just use the original module and add 

prepare_input: fn static = true
prepare_output: fn

and __call__ that basicly call the prepare_input, and prepare_output 
we use maybe_prepare_input():
    if self.prepare_input:
        self.prepare_input(*args, *kwargs)
    else:
        return *rgs, *kwrgs


now please add this to all the nn and layer in models, thank you.

to change this we just need to do dc.replace and change the prepare_input, prepare_output


apply_transforms -> still do a tree traversal do the match operation, 
store the attr, insetad of function to get teh attr, (with getattr(x, attr))
store the cahnge, module, mainly the same module but with prep_input, prep_output

on one go use eqx.tree_at(lambda x: [getattr(x, attr) for attr in store_attr], module, replacement)


remove make_module_opts, for now it's useless


unbox_params on Array means just doing DP, set the PartitionSpec to P()

==============

remove all dtype and param_dtype, we'll other thing to do this
see nn, see models

make the benchmark/bert.py works thankyou.
