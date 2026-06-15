def extract_short_description(
    obj: object, lowercase: bool = True, rstrip_period: bool = True
) -> str:
    if not obj.__doc__:
        raise RuntimeError("obj is missing docstring")
    short_desc = obj.__doc__.split("\n", maxsplit=1)[0]
    if lowercase:
        short_desc = short_desc[0].lower() + short_desc[1:]
    if rstrip_period:
        short_desc = short_desc.rstrip(".")
    return short_desc
