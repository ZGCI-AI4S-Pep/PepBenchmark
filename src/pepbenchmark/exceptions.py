# Copyright ZGCA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Custom Exception Classes for PepBenchmark.

This module defines a hierarchy of custom exception classes to provide
more specific and informative error handling throughout the PepBenchmark
package. Each exception class is designed for specific error scenarios.
"""

from typing import Optional


class PepBenchmarkError(Exception):
    """
    Base exception class for all PepBenchmark-related errors.

    This is the root exception class that all other PepBenchmark exceptions
    inherit from. It provides common functionality for error handling.
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def __str__(self) -> str:
        """Return formatted error message."""
        base_msg = self.message
        if self.error_code:
            base_msg = f"[{self.error_code}] {base_msg}"

        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            base_msg = f"{base_msg} (Details: {detail_str})"

        return base_msg
