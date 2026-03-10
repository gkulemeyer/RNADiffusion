class Registry:
    """
    Generic registry class to map strings (names in config) to classes/functions.
    """

    def __init__(self, name):
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj):
        if name in self._obj_map:
            raise ValueError(f"Object '{name}' is already registered in '{self._name}'.")
        self._obj_map[name] = obj

    def register(self, name=None):
        """
        Decorator to register a class or function.
        Usage:
            @MODEL_REGISTRY.register("my_model")
            class MyModel(nn.Module): ...
        """
        def _register_decorator(obj):
            # If no name is provided, use the class/function name
            register_name = name if name is not None else obj.__name__
            self._do_register(register_name, obj)
            return obj
        return _register_decorator

    def get(self, name):
        """Retrieves the registered object by name."""
        if name not in self._obj_map:
            raise KeyError(
                f"'{name}' not found in registry '{self._name}'. "
                f"Available options: {list(self._obj_map.keys())}"
            )
        return self._obj_map[name]

    def __contains__(self, name):
        return name in self._obj_map

    def __repr__(self):
        return f"Registry(name={self._name}, items={list(self._obj_map.keys())})"


# --- Global Registry Instances ---

# Used in src/models/factory.py to create the model
MODEL_REGISTRY = Registry("Model")

# diffusion schedules
DIFFUSION_REGISTRY = Registry("Diffusion")

# metrics
METRIC_REGISTRY = Registry("Metric")
