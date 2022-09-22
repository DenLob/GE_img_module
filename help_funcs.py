import datetime
import re

try:
    from zoneinfo import ZoneInfo
except ModuleNotFoundError:
    from backports.zoneinfo import ZoneInfo

offset = datetime.timedelta(hours=3)
tz = datetime.timezone(offset, name='МСК')


def date_time_MSC():
    return datetime.datetime.now().astimezone(ZoneInfo(key='Europe/Moscow'))


def date_MSC_str():
    return datetime.datetime.today().astimezone(ZoneInfo(key='Europe/Moscow')).strftime('%Y_%m_%d')
    # return dt


def sort_human(l):
    convert = lambda text: float(text.split('_')[-1].split('.')[0]) if text.split('_')[-1].split('.')[
        0].isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key)]
    l.sort(key=alphanum)
    return l


def path2date_time(path):
    ts = int(path.split('/')[-1].split('_')[0])
    return datetime.datetime.utcfromtimestamp(ts)


def path2str_date(path):
    date_time = path2date_time(path)
    return str(date_time.date().strftime('%Y_%m_%d'))
