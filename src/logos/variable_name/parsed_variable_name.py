"""
Represents a parsed variable name.
"""

from typing import Optional


class ParsedVariableName:
    """
    Performs operations on a string interpreted as a parsed variable name.

    The relevant string format is {template_id}[_{index}].
    """

    def __init__(self, s: str) -> None:
        """
        Initializes a ParsedVariableName object.

        Parameters:
            s: The string interpretation of the parsed variable name.
        """
        toks = s.split("_")
        self._s = s
        self._template_id = toks[0]
        self._index = int(toks[1]) if len(toks) > 1 else -1

    def template_id(self) -> str:
        """
        Returns the template ID of the parsed variable name.

        Returns:
            The template ID of the parsed variable name.
        """
        return self._template_id

    def index(self) -> Optional[int]:
        """
        Returns the index of the parsed variable name.

        Returns:
            The index of the parsed variable name, or None if the index is not
            present.
        """
        return self._index if self._index != -1 else None

    def __str__(self) -> str:
        """
        Returns the string representation of the parsed variable name.

        Returns:
            The string representation of the parsed variable name.
        """
        return self._s
