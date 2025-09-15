import grain



source = grain.MapDataset.source([1,2,3,4,5,6])
loader = grain.DataLoader(data_source = source, sampler = grain.samplers.IndexSampler(len(source), shuffle = True, seed = 53))


iterator = iter(loader)

while True:
    a = next(iterator)
    print(f"DEBUGPRINT[282]: test_is_grain_loader_an_iterator.py:12: a={a}")
