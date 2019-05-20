import sys


def print_progress(iteration, total, prefix='', prog='', round_avg=0, suffix='', time_lapsed=0.0, decimals=1,
                   bar_length=100):
    """
    Call in a loop to create terminal progress bar
    :param iteration: current iteration (Int)
    :param total: total iterations (Int)
    :param prefix: prefix string (Str)
    :param suffix: suffix string (Str)
    :param decimals: positive number of decimals in percent complete (Int)
    :param bar_length: character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = u'\u258B' * filled_length + '-' * (bar_length - filled_length)
    pending_time = (time_lapsed / iteration) * (total - iteration)
    minutes = int(pending_time / 60)
    seconds = round(pending_time % 60)
    suffix = '%d mins, %g secs remaining' % (minutes, seconds)
    sys.stdout.write(
        '\r%s |%s| %s%s - Results page %d of %d - %s - %s' % (
        prefix, bar, percents, '%', iteration, total, prog, suffix))
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
