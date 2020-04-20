import datetime


def str2date(date_str, format="%Y%m%d"):
    return datetime.datetime.strptime(date_str, format)


def date2str(date, format="%Y%m%d"):
    return date.strftime(format)


def int2date_delta(date_delta_int):
    return datetime.timedelta(date_delta_int)