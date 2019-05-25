import datetime
import json
import sys
import time
from config import ROOT_DIR
import os
import re

import requests
from math import floor, ceil
from polyglot.text import Text
import pycountry as country


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
        '\r%s |%s| %s%s - Request %d of %d - %s - %s' % (
            prefix, bar, percents, '%', iteration, total, prog, suffix))
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def identify_language(text):
    """
    Identify language from string using polyglot package
    :param text: String to use for language identification
    :return: Language name (English)
    """
    try:
        if text is not ('' or None):
            text = Text(text)
            language_code = text.language.code
            if language_code is not None:
                language_name = country.languages.get(alpha_2=language_code).name
            else:
                language_name = None
        else:
            language_name = None
            language_code = None
    except AttributeError as ae:
        language_name = None
        language_code = None

    return language_name


def split_time_interval(start, end, intv):
    """
    Split time interval into chunks according to number of sub-intervals specified as intv
    Yields iterable of start/end tuples
    :param start: Start date of time interval
    :param end: End date of time interval
    :param intv: Number of chunks to divide time interval into
    """

    if intv == 1:
        time_delta_days = (end - start).days + 1
        time_delta = end - start
    else:
        time_delta_days = ((end - start).days + 1) / intv  # Search time period length in days
        time_delta = datetime.timedelta(floor(time_delta_days) - 1)

    print('Time delta: %s:' % time_delta)
    print('Days per token list: %d' % time_delta_days)

    for i in range(intv - 1):
        yield (start + (time_delta + datetime.timedelta(days=1)) * i,
               start + (time_delta + datetime.timedelta(days=1)) * i + time_delta)
    yield (start + (time_delta + datetime.timedelta(days=1)) * (intv - 1), end)


def check_access_tokens():
    token_lists = get_access_tokens()
    all_tokens = [token for token_list in token_lists for token in token_list]

    # print(all_tokens)
    query_url = 'https://api.github.com/search/repositories?q=topic:ruby+topic:rails'
    for index, token in enumerate(all_tokens):
        headers = {'Authorization': 'token ' + token}
        response = requests.get(query_url, headers=headers)
        print('\n\nLimit: %d' % int(response.headers['X-RateLimit-Limit']))
        print('Remaining: %d' % int(response.headers['X-RateLimit-Remaining']))
        print('Token: %d,\n%s' % (index, token))


def get_access_tokens():
    # Specify path to credentials
    cred_path = os.path.join(ROOT_DIR, 'DataCollection/credentials')
    # List files with GitHub Access Tokens
    files = [os.path.join(cred_path, f) for f in os.listdir(cred_path) if
             re.match(r'GitHub_Access_Token.*', f)]

    # Initialize empty list of token lists
    token_lists = []

    # Load tokens from text files
    for file in files:
        with open(file, 'r') as f:
            access_tokens = [token.rstrip('\n') for token in f.readlines()]
        # Add token list to list of token lists
        token_lists.append(access_tokens)

    return token_lists
