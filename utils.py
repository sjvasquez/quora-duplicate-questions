def rank(tensor):
    """Get tensor rank as python list"""
    return len(tensor.shape.as_list())


def shape(tensor, dim=None):
    """Get tensor shape/dimension as list/int"""
    if not dim:
        return tensor.shape.as_list()
    if dim:
        return tensor.shape.as_list()[dim]
