import logging
import os
from datetime import datetime, date
from lib.models import LogEntry




class ElasticsearchLogHandler(logging.Handler):

    def __init__(self, level):
        logging.Handler.__init__(self=self)
        #super().__init__(self=self)
        self.setLevel(level)

    def emit(self, record):

        #print(str(record.__dict__), flush=True)

        #{'name': 'werkzeug',
        #  'msg': '192.168.64.1 - - [07/Sep/2024 11:43:23] "%s" %s %s', 
        # 'args': ('GET /socket.io/?EIO=4&transport=websocket&sid=MtyTmZQs5IA6DnvhAAAA HTTP/1.1', '200', '-'), 
        # 'levelname': 'INFO', 
        # 'levelno': 20,
        #  'pathname': '/usr/local/lib/python3.12/dist-packages/werkzeug/_internal.py', 
        # 'filename': '_internal.py', 
        # 'module': '_internal',
        #  'exc_info': None, 
        # 'exc_text': None, 
        # 'stack_info': None, 
        # 'lineno': 97, 
        # 'funcName': '_log',
        #  'created': 1725709403.1972203,
        #  'msecs': 197.0, 
        # 'relativeCreated': 37105.026721954346,
        #  'thread': 133472930760384, 
        # 'threadName': 'Thread-15 (process_request_thread)',
        #  'processName': 'MainProcess',
        #  'process': 26, 
        # 'taskName': None}


        entry = LogEntry(
            message = record.msg,
            level = record.levelname, #record.levelno,
            creation_time = datetime.now(),

            name = record.name,
            pathname = record.pathname,
            filename = record.filename,
            module = record.module,
            lineno = record.lineno,
            funcName = record.funcName,
            threadName = record.threadName,
            processName = record.processName
        )
        entry.save()


def get_log_level(default=logging.WARN):
    LOG_LEVEL = os.getenv("LOG_LEVEL")
    if LOG_LEVEL:
        return eval("logging." + LOG_LEVEL)
    return default




