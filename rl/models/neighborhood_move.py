"""Lightweight move representation shared between data collection and model."""
from dataclasses import dataclass


@dataclass(frozen=True)
class NeighborhoodMove:
    """A generic local-search move.

    Parameters
    ----------
    move_type
        One of "reassign", "swap_machine", "swap_same_machine",
        "swap_order_across".
    op_indices
        Indices of operations involved in the move (schedule order).
    target_machines
        Machines those operations are placed on *after* the move. For moves
        that only reorder operations, this is still the current machines of
        the involved operations, so the model sees the context.
    """

    move_type: str
    op_indices: tuple[int, ...]
    target_machines: tuple[int, ...]
