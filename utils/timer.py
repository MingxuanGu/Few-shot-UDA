from datetime import datetime


def timeit(func):
    """
    A helper function to measure the time elapsed when running some function func.
    Args:
        func: the function to be measured

    Returns:
    The time elapsed.
    """
    def timefunc(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()
        print("{} time elapsed (hh:mm:ss.ms) {}".format(func.__name__, end - start))
        return result
    return timefunc


@timeit
def somefunc():
    result = 1
    for i in range(1, 100000):
        result += i
    return result


class TimeChecker:
    def __init__(self, max_hours=0, max_minutes=0, max_seconds=0):
        """
            save maximum time duration in seconds.
            check whether the program exceed maximum time duration.
            Only used when the device you are running on has a time limit.
        """
        self._max_time_duration = 3600 * max_hours + 60 * max_minutes + max_seconds
        assert self._max_time_duration > 0, 'max time duration should be greater than 0'
        self._time_per_iter = 0
        self._check = None
        self._start_time = None

    def start(self):
        self._start_time = datetime.now()

    def check(self, toprint=False):
        if self._check is None:
            self._check = datetime.now()
            return False
        else:
            now = datetime.now()
            self._time_per_iter = max((now - self._check).seconds, self._time_per_iter)
            if toprint:
                print('time elapsed from start: {}'.format(now - self._start_time))
            return ((now - self._check).seconds + self._time_per_iter) > self._max_time_duration


if __name__ == '__main__':
    import time
    start = datetime.now()
    print("start: {}".format(start))
    # print(somefunc())
    time.sleep(3)
    end = datetime.now()
    print("end: {}".format(end))
    duration = end - start
    print(type(duration))
    print("duration: {}".format(duration))
    print("microseconds: {}".format(duration.microseconds))
    print("days: {}".format(duration.days))
    print(duration.resolution)
    print(duration.seconds)
    print(type(duration.seconds))

