from ctypes import CDLL, c_double, POINTER, c_size_t

FUNCTION_BINDINGS = {
    'garch': {
        'general': {
            'variance': {
                'function_name': 'garch_variance_pq',
                'argtypes': [POINTER(c_double), POINTER(c_double), POINTER(c_double), c_size_t, c_size_t, c_size_t],
                'restype': None
            },
            'likelihood': {
                'function_name': 'normal_likelihood',
                'argtypes': [POINTER(c_double), POINTER(c_double), c_size_t],
                'restype': c_double
            },
            'std_err_robust': {
                'function_name': 'general_garch_pq_std_err_robust',
                'argtypes': [POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_size_t, c_size_t, c_size_t],
                'restype': None
            },
        },
        'special': {
            'objective': {
                'function_name': 'special_garch_oo_normal',
                'argtypes': [POINTER(c_double), POINTER(c_double), POINTER(c_double), c_size_t],
                'restype': c_double
            },
            'variance': {
                'function_name': 'special_garch_oo_normal_variance',
                'argtypes': [POINTER(c_double), POINTER(c_double), POINTER(c_double), c_size_t],
                'restype': None
            },
            'std_err_robust': {
                'function_name': 'special_garch_11_std_err_robust',
                'argtypes': [POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_size_t],
                'restype': None
            },
        }
    },
    'studentt': {
        'general': {
            'likelihood': {
                'function_name': 'any_studentt_likelihood',
                'argtypes': [POINTER(c_double), POINTER(c_double), c_size_t, c_double],
                'restype': c_double
            }
        }
    }
}

class Namespace:
    """A namespace to group related functions or sub-namespaces."""
    def __init__(self):
        self._functions = {}  # Store functions
        self._subnamespaces = {}  # Store nested namespaces

    def register(self, alias, function):
        """Register a function under an alias."""
        self._functions[alias] = function

    def add_namespace(self, name, namespace):
        """Add a sub-namespace."""
        self._subnamespaces[name] = namespace

    def __getattr__(self, name):
        """Allow dot-access for functions or sub-namespaces."""
        if name in self._functions:
            return self._functions[name]
        if name in self._subnamespaces:
            return self._subnamespaces[name]
        raise AttributeError(f"'{name}' not found in this namespace.")

class FunctionBindingManager:
    def __init__(self):
        self.library = CDLL('lib/cvm_lib.so')
        self.bindings = FUNCTION_BINDINGS
        self.namespaces = {}

        self._initialize_bindings()

    def _initialize_bindings(self):
        """Bind functions and organize them into nested namespaces."""
        for namespace, items in self.bindings.items():
            ns = self._create_namespace(items)
            self.namespaces[namespace] = ns

    def _create_namespace(self, items):
        """Recursively create namespaces."""
        ns = Namespace()
        for alias, config in items.items():
            if isinstance(config, dict) and 'function_name' not in config:
                # If the config is a nested dictionary, create a sub-namespace
                sub_ns = self._create_namespace(config)
                ns.add_namespace(alias, sub_ns)
            else:
                # Otherwise, it's a function
                function_name = config['function_name']
                func = getattr(self.library, function_name)
                func.argtypes = config['argtypes']
                func.restype = config['restype']
                ns.register(alias, func)
        return ns

    def __getattr__(self, namespace):
        """Allow dot-access for namespaces."""
        if namespace in self.namespaces:
            return self.namespaces[namespace]
        raise AttributeError(f"Namespace '{namespace}' not found.")

