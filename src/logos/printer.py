from typing import Any
import warnings


class Printer:
    """
    A class for controlling message printing.
    """

    """
    A flag indicating whether or not to print messages to the console.
    """
    SAWMILL_VERBOSE = False

    @classmethod
    def printv(self, msg: Any) -> None:
        """
        Prints a message to the console if in verbose mode.

        Parameters:
            msg: The message to be printed.
        """
        if Printer.SAWMILL_VERBOSE:
            print(msg)

    @classmethod
    def set_verbose(self, val: bool) -> None:
        """
        Sets the verbosity of the printer.

        Parameters:
            val: The new verbosity value.
        """
        Printer.SAWMILL_VERBOSE = val

    @staticmethod
    def set_warnings_to(self, value: str):
        """
        Set selected warnings to `value`.

        Parameters:
            value: The value to set the warnings to.
        """
        warnings.filterwarnings(
            value, category=RuntimeWarning, message="mean of empty slice"
        )
        warnings.filterwarnings(
            value,
            category=RuntimeWarning,
            message="invalid value encountered in scalar divide",
        )
