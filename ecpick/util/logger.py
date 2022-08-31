from datetime import datetime


class Logger:
    FORMAT = "{datetime:%Y-%m-%d %I:%M:%S} {type:>6} : {msg}"
    FILE = None
    CONSOLE = True

    @classmethod
    def init(cls, path, console):
        cls.FILE = open(path, 'w+')
        cls.CONSOLE = console

    @classmethod
    def log(cls, msg, level):
        msg = msg.split("\n")

        for m in msg:
            formatted_msg = cls.FORMAT.format(type=level, datetime=datetime.now(), msg=m)

            if cls.CONSOLE:
                print(formatted_msg)

            if cls.FILE is not None:
                cls.FILE.write(formatted_msg + "\n")
                cls.FILE.flush()

    @classmethod
    def info(cls, msg):
        cls.log(msg, "INFO")

    @classmethod
    def trace(cls, msg):
        cls.log(msg, "TRACE")

    @classmethod
    def debug(cls, msg):
        cls.log(msg, "DEBUG")

    @classmethod
    def warn(cls, msg):
        cls.log(msg, "WARN")

    @classmethod
    def error(cls, msg):
        cls.log(msg, "ERROR")
