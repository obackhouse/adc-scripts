import importlib, os

files = os.listdir(os.path.dirname(os.path.realpath(__file__)))
__all__ = []

for f in files:
    if 'adc' in f and '.py' in f:
        name = f[:-3]
        globals()[name] = importlib.import_module('adc2.%s' % name)
        __all__.append(name)
