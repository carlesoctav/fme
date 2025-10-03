class ContextParallelAttention(eqx.Module):
    base: nn.AttentionModule
    axis_name: str = eqx.field(static=True)
    mesh: Mesh = eqx.field(static=True)

    def __call__(
        self,
        q,
        k,
        v,
        *,
        dropout,
        attention_mask=None,
        segment_ids=None,
        key=None,
    ):
        axis_name = self.axis_name
        mesh = self.mesh

        q_t = jnp.moveaxis(q, -3, 0)
        k_t = jnp.moveaxis(k, -3, 0)
        v_t = jnp.moveaxis(v, -3, 0)

        @ft.partial(
            shard_map,
            in_specs=(P(axis_name), P(axis_name), P(axis_name), None, None, None, None),
            out_specs=P(axis_name),
            mesh=mesh,
        )
        def shard_apply(q_local, k_local, v_local, mask_arg, seg_arg, dropout_arg, key_arg):
            axis_index = jax.lax.axis_index(axis_name)
            axis_size = jax.lax.psum(1, axis_name=axis_name)

            local_q = jnp.moveaxis(q_local, 0, -3)
            local_k = jnp.moveaxis(k_local, 0, -3)
            local_v = jnp.moveaxis(v_local, 0, -3)

            gathered_k = jax.lax.all_gather(k_local, axis_name=axis_name, axis=0, tiled=False)
            gathered_v = jax.lax.all_gather(v_local, axis_name=axis_name, axis=0, tiled=False)

            gathered_k = gathered_k.reshape(
                (axis_size * k_local.shape[0],) + k_local.shape[1:]
            )
            gathered_v = gathered_v.reshape(
                (axis_size * v_local.shape[0],) + v_local.shape[1:]
            )

            global_k = jnp.moveaxis(gathered_k, 0, -3)
            global_v = jnp.moveaxis(gathered_v, 0, -3)

            local_q_len = local_q.shape[-3]
            start = axis_index * local_q_len

            mask = None
            if mask_arg is not None:
                mask = jnp.asarray(mask_arg)
                if mask.ndim == 2:
                    mask = jax.lax.dynamic_slice(
                        mask,
                        (0, start),
                        (mask.shape[0], local_q_len),
                    )
                elif mask.ndim == 4:
                    mask = jax.lax.dynamic_slice(
                        mask,
                        (0, 0, start, 0),
                        (
                            mask.shape[0],
                            mask.shape[1],
                            local_q_len,
                            mask.shape[3],
                        ),
                    )
                else:
                    raise ValueError(
                        "context_parallel only supports attention masks with 2 or 4 dimensions"
                    )

            segments = None
            if seg_arg is not None:
                seg = jnp.asarray(seg_arg)
                segments = jax.lax.dynamic_slice(
                    seg,
                    (0, start),
                    (seg.shape[0], local_q_len),
                )

            shard_key = None
            if key_arg is not None:
                shard_key = jax.random.fold_in(key_arg, axis_index)

            out = self.base(
                local_q,
                global_k,
                global_v,
                dropout=dropout_arg,
                attention_mask=mask,
                segment_ids=segments,
                key=shard_key,
            )

            return jnp.moveaxis(out, -3, 0)

        result = shard_apply(
            q_t,
            k_t,
            v_t,
            attention_mask,
            segment_ids,
            dropout,
            key,
        )
        return jnp.moveaxis(result, 0, -3)
