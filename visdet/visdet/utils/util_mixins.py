# ruff: noqa
# type: ignore
# Simple util mixins


class NiceRepr:
    """Inherited by objects that have a nice string representation.

    Define ``__nice__`` to get ``__repr__`` and ``__str__`` for free.
    """

    def __nice__(self):
        """Returns a nice string representation of the object."""
        return ""

    def __repr__(self):
        """Return the nice string representation."""
        try:
            nice = self.__nice__()
            classname = self.__class__.__name__
            return f"<{classname}({nice})>"
        except Exception:
            return object.__repr__(self)

    def __str__(self):
        """Return the nice string representation."""
        return self.__repr__()
