"""Minimal logger stub for evaluation metrics borrowed from POEM-v2."""

from __future__ import annotations

import logging

_log = logging.getLogger("eval.metrics")


class _MetricLogger:
    """Thin wrapper so metrics can call ``logger.error`` / ``logger.debug``."""

    def error(self, msg, *args, **kwargs):
        _log.error(msg, *args, **kwargs)
        return msg

    def debug(self, msg, *args, **kwargs):
        _log.debug(msg, *args, **kwargs)
        return msg

    def info(self, msg, *args, **kwargs):
        _log.info(msg, *args, **kwargs)
        return msg

    def warning(self, msg, *args, **kwargs):
        _log.warning(msg, *args, **kwargs)
        return msg


logger = _MetricLogger()
