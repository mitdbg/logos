class UnsupportedOperationError(RuntimeError):
    """
    Raised when an operation is not supported for the current LOGos
    entry point (e.g. calling parse() on an instance created from a
    pre-parsed table).
    """
