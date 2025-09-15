i want you create the logger logic


the logger takes
step_interval: params


log_step(current_step, metrics, namesspace = "eval") -> to log everystep interval
     if not curren_step mod step_interval:
         return

    metric_with_namespace = just lopping and prepend eval 
    log(metrics_with_namespace)


Loss(Metric)
    counts: jnp.array(int)
    values: jnp.array(flaot)
    mode: list [literal["mean", "max", "min", "count", "std"]]

    update(**kwargs)-> self:
    self.values = new_array(self.values, kwrgs["values"])
    if kwargs["count]:
        self.count = new_array(self.values, kwrgs["count"])
    else
        self.count = kwerg["value].size

    (eqx.tree_at([values, count], r, [values+new_value, count+new_count | new_value.size]))
    
    i think you need to use eqax.tree_at here :D
    return new_self

example aux from loss_function
aux = {
    loss = Loss(value, mode = "mean")
    xx = TokenPerSec()
    aa = Accuracy(value, mode = "max")
}


update_metrics_from_aux(aux, metrics):
    def _f(aux_leaf, metric_leaf):
            return metric_leaf.updates(*aux_leaf)

    filtered_aux_that_is_metric = filter_aux(aux_leaf)
    jax.tree_map_util(_f, fiiltered_aux_that_is_metric, metrics, is_leaf = is_metric_leaf)


and then


self._logger.log_step(sefl.step, step_metrics)
inside this log_step there's 
reduced_metrics = reduce_metrics(step_metrics)
to reduce the metrics based on the mode
    
