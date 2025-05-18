import json
import yaml
from pathlib import Path


class Config(dict):
    """A minimal replacement for ``mmcv.Config``."""

    @classmethod
    def fromfile(cls, filename):
        filename = Path(filename)
        if filename.suffix in {'.yml', '.yaml'}:
            with open(filename, 'r') as f:
                data = yaml.safe_load(f)
        elif filename.suffix == '.json':
            with open(filename, 'r') as f:
                data = json.load(f)
        elif filename.suffix == '.py':
            cfg_dict = {}
            with open(filename, 'r') as f:
                code = compile(f.read(), str(filename), 'exec')
                exec(code, {}, cfg_dict)
            data = {k: v for k, v in cfg_dict.items() if not k.startswith('_')}
        else:
            raise OSError(f'Unsupported config type: {filename}')
        return cls(data)

    def dump(self, file):
        file = Path(file)
        if file.suffix in {'.yml', '.yaml'}:
            with open(file, 'w') as f:
                yaml.safe_dump(dict(self), f)
        elif file.suffix == '.json':
            with open(file, 'w') as f:
                json.dump(self, f)
        else:
            raise OSError(f'Unsupported dump type: {file}')
