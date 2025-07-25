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

import logging
import os

_LOGGER = None


def get_logger(name="pepbenchmark"):
    """Get or create a logger for the whole project."""
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    logger = logging.getLogger(name)
    log_level = os.environ.get("PEPBENCHMARK_LOGLEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s][%(levelname)s][%(name)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    _LOGGER = logger
    return logger


def set_log_level(level: str):
    """Set log level for the pepbenchmark logger."""
    logger = get_logger()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))


def enable_logging():
    set_log_level("INFO")


def disable_logging():
    set_log_level("CRITICAL")
