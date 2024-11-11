"""
Represents a prepared variable name.
"""

from typing import Optional, Self, Union

from src.logos.variable_name.parsed_variable_name import ParsedVariableName


class PreparedVariableName:
    """
    Performs operations on a string interpreted as a prepared variable name.

    The relevant string format is
    {template_id}[_{index}][={pre-agg value}]+{aggregate}[={post_agg value}].
    """

    def __init__(self, s: str) -> None:
        """
        Initializes a PreparedVariableName object.

        Parameters:
            s: The string representation of the prepared variable name.
        """
        mid_split = s.split("+")

        left_split = mid_split[0].split("=")
        right_split = mid_split[1].split("=") if len(mid_split) > 1 else ["", ""]

        self._base_var = left_split[0]
        self._pre_agg_value = left_split[1] if len(left_split) > 1 else ""
        self._aggregate = right_split[0]
        self._post_agg_value = right_split[1] if len(right_split) > 1 else ""

    def base_var(self) -> str:
        """
        Returns the base variable of the prepared variable name.

        Returns:
            The base variable of the prepared variable name.
        """
        return self._base_var

    def template_id(self) -> str:
        """
        Returns the template ID of the prepared variable name. If the base variable
        is 'TemplateId', then this will match the pre_agg_value.

        Returns:
            The template ID of the prepared variable name.
        """
        if self._base_var == "TemplateId":
            return self._pre_agg_value
        else:
            return ParsedVariableName(self._base_var).template_id()

    def index(self) -> Optional[int]:
        """
        Returns the index of the prepared variable name.

        Returns:
            The index of the prepared variable name, or None if the index is not
            present.
        """
        return ParsedVariableName(self._base_var).index()

    def pre_agg_value(self) -> str:
        """
        Returns the pre-aggregate value of the prepared variable name.

        Returns:
            The pre-aggregate value of the prepared variable name.
        """
        return self._pre_agg_value

    def aggregate(self) -> str:
        """
        Returns the aggregate of the prepared variable name.

        Returns:
            The aggregation function implied by the prepared variable name.
        """
        return self._aggregate

    def post_agg_value(self) -> str:
        """
        Returns the post-aggregate value of the prepared variable name.

        Returns:
            The post-aggregate value of the prepared variable name.
        """
        return self._post_agg_value

    def no_pre_post_aggs(self) -> bool:
        """
        Check whether the prepared variable has no pre- or post-aggregates.

        Returns:
            Whether the prepared variable has no pre- or post-aggregates.
        """
        return self.pre_agg_value() == "" and self.post_agg_value() == ""

    def has_base_var(self, x: Union[str, Self]) -> bool:
        """
        Check whether the prepared variable has the given base variable.

        Parameters:
            x: The base variable to check.

        Returns:
            Whether the prepared variable has the given base variable.
        """
        return PreparedVariableName.same_base_var(self, x)

    @staticmethod
    def same_base_var(
        var1: Union[str, "PreparedVariableName"],
        var2: Union[str, "PreparedVariableName"],
    ) -> bool:
        """
        Check whether two prepared variables have the same base variable.

        Parameters:
            var1: The first variable to check.
            var2: The second variable to check.

        Returns:
            Whether the two variables have the same base variable.
        """

        if isinstance(var1, str):
            var1 = PreparedVariableName(var1)
        if isinstance(var2, str):
            var2 = PreparedVariableName(var2)

        return var1.base_var() == var2.base_var()
