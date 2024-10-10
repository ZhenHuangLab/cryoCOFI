import os

def get_lib_path(lib_name):
    '''
    Get the path to the library.
    Args:
        lib_name: name of the library
    Returns:
        path: path to the library
    '''
    return os.path.join(os.path.dirname(__file__), 'lib', f'{lib_name}.so')