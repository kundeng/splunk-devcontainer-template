# some basic utils for dev

import os


def format_float_list(in_list):
    return ["%.3f" % s for s in in_list]


def str_int_list_no_bracket(int_list):
    return ",".join(map(str, int_list))


def show_anomaly_figure():
    return _check_env_variable("SHOW_ANOMALY_FIG")


def print_pattern_scores():
    return _check_env_variable("PRINT_PATTERN_SCORES")


def is_verbose():
    return _check_env_variable("TIMEPOLICY_VERBOSE")


def show_anomaly_score():
    return _check_env_variable("SHOW_ANOMALY_SCORE")


def disable_multi_resolution():
    return _check_env_variable("DISABLE_MULTI_RESOLUTION")


def _check_env_variable(var_name):
    v = os.getenv(var_name)
    if v:
        return v.lower() == "true"
    return False


def find_consecutive_ones(series):
    cnt = series[0]
    for i in range(1, len(series)):
        if series[i] == 1 and series[i] != series[i - 1]:
            cnt += 1
    return cnt
