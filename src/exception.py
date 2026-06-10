"""
exception.py

Custom exception handling for MLOps pipeline.

Improves debugging by:
- capturing file name
- capturing line number
- providing structured error message


"""

import sys


def error_message_detail(error, error_detail: sys):
    """
    Extracts detailed error information.

    Parameters
    ----------
    error : Exception
        The exception object
    error_detail : sys
        sys module for traceback info

    Returns
    -------
    str
        Formatted error message
    """

    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    error_message = (
        f"Error occurred in script: [{file_name}] "
        f"at line number: [{line_number}] "
        f"error message: [{str(error)}]"
    )

    return error_message


# =========================
# CUSTOM EXCEPTION CLASS
# =========================
class CustomException(Exception):
    """
    Custom exception class for ML pipeline errors.
    """

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)

        self.error_message = error_message_detail(
            error_message,
            error_detail
        )

    def __str__(self):
        return self.error_message