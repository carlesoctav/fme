def f(*args):
    print(type( args ))
    return args[0]



return_value = f(1,2,2,3,3)
print(f"DEBUGPRINT[240]: args_is_always_tuple.py:7: return_value={type( return_value )}")
