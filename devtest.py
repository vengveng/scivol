import volkit, inspect, pprint

# 1) Quick list of all public names (skip dunders/private helpers)
# ['any_studentt_likelihood',
pprint.pp([name for name in dir(volkit) if not name.startswith("_")])
#  'garch_variance_pq',
#  'general_garch_pq_std_err_robust',
#  'normal_likelihood',
#  'special_garch_11_std_err_robust',
#  'special_garch_oo_normal',
#  'special_garch_oo_normal_variance']

# 2) Look at a single function’s signature and docstring
# help(volkit.normal_likelihood)
# or
# print(inspect.signature(volkit.normal_likelihood))
#    (sigma2_ptr, eps2_ptr, n)

# 3) Where is the compiled shared object located?
print(volkit.__file__)