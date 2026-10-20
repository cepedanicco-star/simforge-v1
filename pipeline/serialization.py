"""Convert native annotation records without changing values or inventing labels."""


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, 'tolist'):
        return jsonable(value.tolist())
    return value
