# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import os

import numpy as np
from onnxruntime.capi import _pybind_state as C


class CheckpointState:
    """Class that holds the state of the training session

    This class holds all the state information of the training session such as the model parameters,
    its gradients, the optimizer state and user defined properties.

    User defined properties can be indexed by name from the `CheckpointState` object.

    To create the `CheckpointState`, use the `CheckpointState.load_checkpoint` method.

    Args:
        state: The C.Checkpoint state object that holds the underlying session state.
    """

    def __init__(self, state: C.CheckpointState):
        if not isinstance(state, C.CheckpointState):
            raise TypeError(f"Invalid argument for CheckpointState received {type(state)}")
        self._state = state

    @classmethod
    def load_checkpoint(cls, checkpoint_uri: str | os.PathLike) -> CheckpointState:
        """Loads the checkpoint state from the checkpoint file

        Args:
            checkpoint_uri: The path to the checkpoint file.

        Returns:
            CheckpointState: The checkpoint state object.
        """
        return cls(C.load_checkpoint(os.fspath(checkpoint_uri)))

    @classmethod
    def save_checkpoint(
        cls, state: CheckpointState, checkpoint_uri: str | os.PathLike, include_optimizer_state: bool = False
    ) -> None:
        """Saves the checkpoint state to the checkpoint file

        Args:
            state: The checkpoint state object.
            checkpoint_uri: The path to the checkpoint file.
            include_optimizer_state: If True, the optimizer state is also saved to the checkpoint file.
        """
        C.save_checkpoint(state._state, os.fspath(checkpoint_uri), include_optimizer_state)

    def __getitem__(self, name: str) -> int | float | str | np.ndarray:
        """Gets the property associated with the given name

        Args:
            name: The name of the property

        Returns:
            The value of the property
        """
        if self._state.has_property(name):
            return self._state.get_property(name)
        elif self._state.has_parameter(name):
            return self._state.get_parameter_data(name)

    def __setitem__(self, name: str, value: int | float | str | np.ndarray) -> None:
        """Sets the property value for the given name

        Args:
            name: The name of the property
            value: The value of the property
        """
        if self._state.has_property(name):
            self._state.add_property(name, value)
        elif self._state.has_parameter(name):
            self._state.update_parameter_data(name, value)

    def __contains__(self, name: str) -> bool:
        """Checks if the property exists in the state

        Args:
            name: The name of the property

        Returns:
            True if the property exists, False otherwise
        """
        return self._state.has_property(name) or self._state.has_parameter(name)
