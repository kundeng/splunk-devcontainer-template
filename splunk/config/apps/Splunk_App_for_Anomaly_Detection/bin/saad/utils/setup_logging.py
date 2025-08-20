#!/usr/bin/env python

import os
import logging
import logging.handlers
import html

BASE_LOGGER_NAME = "anomalyapp"  # Name of the log file (i.e. `anomalyapp.log`) to which all CSCs will write their telemetry
DEFAULT_LEVEL = logging.INFO
ANOMALY_APP_TELEMETRY = "AnomalyApp Telemetry:"


def get_splunkhome_path():
    return os.path.normpath(os.environ["SPLUNK_HOME"])


def make_splunkhome_path(p):
    return os.path.join(get_splunkhome_path(), *p)


def sanitize_log_input(input_string):
    # Comprehensive sanitization to prevent log injection attacks
    return html.escape(input_string)


def get_logger(name=BASE_LOGGER_NAME, level=DEFAULT_LEVEL):
    """Returns a general-purpose logger instance.

    The logger is configured to write to both:
      * A (rotated) file in $SPLUNK_HOME/var/log/splunk/<name>.log
      * Standard error.

    Additionally, it consults $SPLUNK_HOME/etc/log.cfg and
    log-local.cfg for default log-levels. You can configure per-logger
    log-levels by adding a property to log-local.cfg that looks like:

        [python]
        myloggername = DEBUG

    For DEBUG messages to show up in search.log as well, you will need
    to modify $SPLUNK_HOME/etc/log-searchprocess-local.cfg to contain:

        category.ChunkedExternProcessor=DEBUG

    Idiomatic usage is:

        #!/usr/bin/env python
        import setup_logging
        logger = setup_logging.get_logger()

        def foo():
            logger.warn("Red Alert, report to battle stations")

    """
    logger = logging.getLogger(name)

    # Initial setup
    if len(logger.handlers) == 0:
        logger.setLevel(level)
        logger.propagate = False

        has_splunk_home = os.environ.get("SPLUNK_HOME")
        if has_splunk_home:
            path = make_splunkhome_path(["var", "log", "splunk", name + ".log"])
            backup_count = 5
        else:
            # No backups if logging to the current directory
            path = os.path.normpath(os.path.join(os.getcwd(), name + ".log"))
            backup_count = 0

        try:
            file_handler = logging.handlers.RotatingFileHandler(
                path, maxBytes=1000000, backupCount=backup_count
            )
            formatter = logging.Formatter(
                "%(created)f PID %(process)d %(asctime)s %(levelname)s [%(name)s] [%(funcName)s] %(message)s"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except (FileNotFoundError, PermissionError) as e:
            logger.error(f"Error while setting up log file: {str(e)}")

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logger.addHandler(stream_handler)

        if has_splunk_home:
            try:
                import splunk

                # Read logging level information from log.cfg so it will overwrite log
                # Note if logger level is specified on that file then it will overwrite log level
                LOGGING_DEFAULT_CONFIG_FILE = make_splunkhome_path(["etc", "log.cfg"])
                LOGGING_LOCAL_CONFIG_FILE = make_splunkhome_path(
                    ["etc", "log-local.cfg"]
                )
                LOGGING_STANZA_NAME = "python"
                splunk.setupSplunkLogger(
                    logger,
                    LOGGING_DEFAULT_CONFIG_FILE,
                    LOGGING_LOCAL_CONFIG_FILE,
                    LOGGING_STANZA_NAME,
                    verbose=False,
                )
            except ImportError:
                logger.warn(
                    "Unable to import splunk python module: cannot set up splunk logging. "
                    "If you are using the AnomalyApp's code directly, "
                    "without a custom search command, you may see this warning."
                )
        else:
            logger.warn("No SPLUNK_HOME set. Logging to %s", sanitize_log_input(path))

    return logger
