""" This code file was adapted from the Detectron2 project """

import sys
import functools
import logging
import termcolor
import inspect
from pathlib import Path

# For type annotations
from typing import Optional
from logging import Logger


class ColorfulFormatter(logging.Formatter):
    def __init__(self, root_name: str, abbr_name: Optional[str] = None, *args, **kwargs):
        # self._root_name = str(kwargs.get("root_name", "")) + "."
        self._root_name = root_name
        self._abbr_name = abbr_name
        # if self._abbrev_name is not None:
        #     self._abbrev_name = self._abbrev_name + "."
        super(ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        if self._abbr_name is not None:
            record.name = record.name.replace(self._root_name, self._abbr_name)

        log = super(ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = termcolor.colored("WARNING", "yellow", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = termcolor.colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


def log(msg: str, logger: Optional[Logger], log_type: str = "info"):
    """ Helper function to log if a logger is provided """
    if isinstance(logger, Logger):
        if log_type == "info":
            logger.info(msg)
        if log_type == "warn":
            logger.warn(msg)
        if log_type == "error":
            logger.error(msg)


# so that calling setup_logger multiple times won't add many handlers
@functools.lru_cache()
def setup_logger(dist_rank: int, name: str | None = None, abbr_name: str | None = None, output: str | Path | None = None, color: bool = True) -> Logger:
    """
    Initialize logger and set its verbosity level to "DEBUG".

    Args:
        dist_rank: The rank from which this function is being called.
                   If the rank is 0, the logger will be set to log to
                   the standard output in addition to the {output}
                   file/directory. If the rank is other than 0, this
                   function will just set the logger to log to the
                   {output} file/directory for the given rank.
        name:      The name of the logger.
        abbr_name: The abbreviated name of the logger.
        output:    A file name or a directory to save log.
                   If {output} is None, the log file won't be saved.
                   If {output} is a directory the logger's logs will
                   be saved to "{output}/log.txt" file. Otherwise,
                   logs will be saved to {output}.
        color:     If the logger's logs should be colored in the
                   standard output.

    Return:
        A logging Logger object.
    """
    if name is None:
        name = inspect.stack()[-1].filename
        name = Path(name).name

    if not isinstance(output, Path) and output is not None:
        output = Path(output)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # stdout logging: master only
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(
        "[%(asctime)s %(name)s]: %(message)s",
        datefmt=date_format
    )
    if dist_rank == 0:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(logging.DEBUG)
        if color:
            handler.setFormatter(ColorfulFormatter(
                    name,
                    abbr_name,
                    fmt=f"{termcolor.colored('[%(asctime)s %(name)s]: ', 'green')}%(message)s",
                    datefmt=date_format,
                )
            )
        else:
            handler.setFormatter(formatter)
        logger.addHandler(handler)

    # file logging: all workers
    if output is not None:
        if output.is_dir():
            output_path = output
            filename = str(output.joinpath("log.txt"))
        else:
            output_path = output.parent
            filename = str(output)

        output_path.mkdir(parents=True, exist_ok=True)
        if dist_rank != 0:
            filename = str(filename) + f"_rank_{dist_rank}"

        handler = logging.StreamHandler(_cached_log_stream(filename))
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return open(filename, "a")
