we need to test each high level model on ./models
for examples: BertModel, BertForMaskedLM (see the ./models/bert/__init__.py to check which one is the high level models)
here's the thing you need to test
1. equivalance of the output (given this implementation and transformers)
2. initialize module under make_module_opt, convert the abstract_module -> module (need to there's no shapedtypestruct)

please revisit current test and simplify this thing
you can actually create some utils on tests/utils.py

