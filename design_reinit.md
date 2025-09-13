let's say we've an abstarct_module which was computed something liek this, eqx.filter_eval_shape(Linear, args, kwargs)
(the main different is .value was replaced by fake flops, aka jax.dtypestruct)


and then i do apply_transforms on this abstract_module to do parallelism based on the plan see ./temp/my_great_tp_plan.py

then this abstract_module transofrms ped to another abstarct_module with a new Darray value (add pspec) and also add forward_pre_hook and forward_hook (see make_module_with_sharded_constraint)


the functinality that's missing here is to reinit the value (make a real array and real module) from the abstract_module


my main_idea was we store the initalizer of (atomic module, like linear, etc), and we just using iter_module, reinit every



Plan: dict[[Path (glob style), param], Initalizer | None]


def init_weights(module, initalizer):
    new_weights = initalizer(module, initalizer)

    if under_mesh context amanger or jax.mesh is set and module.weight
        if pspec = jax.NamedSharding(new_weights, mesh, modu) 


re_init_weights(module, Optional[Plan]):
    replacement:
    for path, module in iter_module():
        if atomic_module(module):
            new_init = None
            if match(path, plan):
                new_init = plan[path]
            else:
                if not hasattr(module, initalizer):
                    raise valueError
                new_init = getattr(module, initalizer) 

            module_with_weights = init_weights(module, new_init) 
            replacement.append(module_with_weights)
            where.append(klajf;lksdjf)

    return eqx.tree_at(where, module, replacement)

