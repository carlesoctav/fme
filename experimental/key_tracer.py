import jax


@jax.jit
def init(key):
    rand_element = jax.random.normal(key, (10, 10))
    print(f"DEBUGPRINT[52]: key_tracer.py:6: rand_element={rand_element}")
    return rand_element





hlo = init.lower(jax.random.key(10)).as_text()
print(f"DEBUGPRINT[53]: key_tracer.py:14: hlo={hlo}")
