from datetime import datetime


def record_time(record_list, func, args, time_format="%Y%m%d-%H:%M:%S"):
    start_time = datetime.now()
    result = func(*args)
    end_time = datetime.now()
    duration = end_time - start_time
    record_list.append([
        start_time.strftime(time_format),
        end_time.strftime(time_format),
        format_timedelta(duration),
    ])
    return result

def format_timedelta(timedelta):
    total_seconds = int(timedelta.total_seconds())
    hours, remainder = divmod(total_seconds, 60*60)
    minutes, seconds = divmod(remainder, 60)
    return "{}:{}:{}".format(hours, minutes, seconds)


