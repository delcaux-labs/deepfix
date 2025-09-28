"""Custom exceptions for Cursor CLI integration."""


class CursorError(Exception):
    """Base exception for Cursor CLI integration errors."""

    pass


class CursorTimeoutError(CursorError):
    """Raised when Cursor CLI operation times out."""

    pass


class CursorNotFoundError(CursorError):
    """Raised when Cursor CLI is not found or not installed."""

    pass


class CursorAuthenticationError(CursorError):
    """Raised when authentication with Cursor fails."""

    pass


class CursorResponseError(CursorError):
    """Raised when there's an error in the response from Cursor CLI."""

    pass
