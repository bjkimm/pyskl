class Registry:
    """A very simple registry to map strings to classes."""

    def __init__(self, name, parent=None):
        self._name = name
        self._module_dict = {}
        self._parent = parent

    def register_module(self, cls=None, name=None):
        if cls is None:
            def _register(cls):
                self._register(cls, name)
                return cls
            return _register
        self._register(cls, name)
        return cls

    def _register(self, cls, name=None):
        reg_name = name or cls.__name__
        if reg_name in self._module_dict:
            raise KeyError(f'{reg_name} is already registered in {self._name}')
        self._module_dict[reg_name] = cls

    def get(self, key):
        if self._parent and key in self._parent._module_dict:
            return self._parent._module_dict[key]
        return self._module_dict.get(key)

    def build(self, cfg, default_args=None):
        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()
        obj_type = args.pop('type')
        if isinstance(obj_type, str):
            obj_cls = self.get(obj_type)
            if obj_cls is None:
                raise KeyError(f'{obj_type} is not registered in {self._name}')
        else:
            obj_cls = obj_type
        if default_args:
            for name, value in default_args.items():
                args.setdefault(name, value)
        return obj_cls(**args)
