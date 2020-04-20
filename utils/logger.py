import time
import logging
from colorama import Fore, Back, Style


class ColoredFormatter(logging.Formatter):
    def __init__(self):
        self._fmt = "{prefix_style}[{levelname:.1} {asctime}]{reset} {message_style}{message}{reset}"
        self._style = "{"
        super().__init__(fmt=self._fmt, datefmt="%Y-%m-%d %H:%M:%S", style=self._style)
        self._pallet = {
            "CRITICAL": Fore.BLUE,
            "ERROR": Fore.RED,
            "WARNING": Fore.YELLOW,
            "INFO": Fore.GREEN,
            "DEBUG": Fore.MAGENTA,
        }

    def format(self, record):
        record.prefix_style = self._pallet[record.levelname]
        record.message_style = self._pallet[record.levelname]+Style.BRIGHT
        record.reset = Style.RESET_ALL
        return logging.Formatter.format(self, record)


class PrettyLogger(object):
    """Customized logger for pretty logging messages"""

    def __init__(self, level=logging.DEBUG):
        self._level = level
        self.logger = self._init_logger()

    def get_logger(self):
        return self.logger

    def _init_logger(self, clear_prev_handlers=True):
        logger = logging.getLogger()
        logger.setLevel(self._level)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self._level)
        colored_formatter = ColoredFormatter()
        console_handler.setFormatter(colored_formatter)
        if clear_prev_handlers:
            logger.handlers.clear()
        logger.addHandler(console_handler)
        return logger
    
    def _join_words(self, words):
        return " ".join(map(str, words))

    def critical(self, *words):
        self.logger.critical(self._join_words(words))

    def error(self, *words):
        self.logger.error(self._join_words(words))

    def warning(self, *words):
        self.logger.warning(self._join_words(words))

    def info(self, *words):
        self.logger.info(self._join_words(words))

    def debug(self, *words):
        self.logger.debug(self._join_words(words))