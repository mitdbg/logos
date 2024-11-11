from typing import Any


class Printer:
    """
    A class for controlling message printing.
    """

    """
    A flag indicating whether or not to print messages to the console.
    """
    LOGOS_VERBOSE = False

    @classmethod
    def printv(self, msg: Any) -> None:
        """
        Prints a message to the console if in verbose mode.

        Parameters:
            msg: The message to be printed.
        """
        if Printer.LOGOS_VERBOSE:
            print(msg)

    @classmethod
    def set_verbose(self, val: bool) -> None:
        """
        Sets the verbosity of the printer.

        Parameters:
            val: The new verbosity value.
        """
        Printer.LOGOS_VERBOSE = val
