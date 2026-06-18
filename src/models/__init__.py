"""Model definitions."""

__all__ = ["PlaceholderGNN"]


def __getattr__(name: str):
    if name == "PlaceholderGNN":
        from src.models.gnn_model import PlaceholderGNN

        return PlaceholderGNN
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
