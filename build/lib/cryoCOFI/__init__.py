import os

def get_lib_path(lib_name):
    return os.path.join(os.path.dirname(__file__), 'lib', f'{lib_name}.so')