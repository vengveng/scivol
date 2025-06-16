import volkit, pprint
pprint.pp([n for n in dir(volkit) if not n.startswith('_')])
print("C extension path:", volkit._core.__file__)