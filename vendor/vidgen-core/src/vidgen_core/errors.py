"""Domain errors. Class names matter: Temporal matches them in non_retryable_error_types."""


class VidgenError(Exception):
    """Base class for all vidgen errors."""


class ModerationBlocked(VidgenError):  # noqa: N818
    """Provider refused the content; retrying will not help."""


class InvalidRequest(VidgenError):  # noqa: N818
    """Request is malformed or unsupported; retrying will not help."""


class BudgetExceeded(VidgenError):  # noqa: N818
    """Estimated cost exceeds the remaining budget."""

    def __init__(self, required_usd: float, remaining_usd: float) -> None:
        super().__init__(
            f"Budget exceeded: need {required_usd:.2f} USD, remaining {remaining_usd:.2f} USD"
        )
        self.required_usd = required_usd
        self.remaining_usd = remaining_usd


class NotFound(VidgenError):  # noqa: N818
    """Requested entity does not exist."""


class StorageError(VidgenError):
    """Asset storage failure."""
