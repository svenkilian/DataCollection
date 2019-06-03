# This module implements helper functions providing ancillary functionality to other modules and functions

import datetime
import sys
import time
from multiprocessing import current_process
from bs4 import BeautifulSoup
from config import ROOT_DIR
import os
import re
from math import floor
from polyglot.text import Text
import pycountry as country


def print_progress(iteration, total, prefix='', prog='', round_avg=0, suffix='', time_lapsed=0.0, decimals=1,
                   bar_length=100):
    """
    Creates terminal progress bar by being called in a loop.

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
        '\r%s |%s| %s%s - Request %d of %d - %s - %s - %s' % (
            prefix, bar, percents, '%', iteration, total, current_process().name, prog, suffix))
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def identify_language(text):
    """
    Identifies language from string using polyglot package.

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


def split_time_interval(start, end, intv, n_days):
    """
    Splits time interval into chunks according to number of sub-intervals specified as intv.
    Yields iterable of start/end tuples.

    :param start: Start date of time interval
    :param end: End date of time interval
    :param intv: Number of chunks to divide time interval into
    """

    if n_days > 1:
        n_days_micro = floor(n_days / intv)  # Micro search time period length in days
        time_delta = datetime.timedelta(days=n_days_micro - 1)
        # Yield start and end date for time period
        for i in range(intv - 1):
            yield (start + (time_delta + datetime.timedelta(days=1)) * i,
                   start + time_delta + (time_delta + datetime.timedelta(days=1)) * i)
        # Yield last time period
        yield (start + (time_delta + datetime.timedelta(days=1)) * (intv - 1), end)

    else:
        n_days_micro = n_days
        time_delta = end - start
        yield (start, end)

    print('Time delta per time frame: %s:' % time_delta)
    print('Days per time frame: %d' % n_days_micro)


def check_access_tokens(token_index, response):
    """
    Checks state of access tokens and prints state in console; Pauses calling thread if limit is sufficiently low.

    :param token_index: Index of token currently in use
    :param response: Response object from last API request
    :return:
    """
    try:
        print('\n\nRemaining/Limit for token %d: %d/%d' % (token_index,
                                                           int(response.headers['X-RateLimit-Remaining']),
                                                           int(response.headers['X-RateLimit-Limit'])))
        if int(response.headers['X-RateLimit-Remaining']) <= 6:
            time.sleep(15)
            print('Execution paused for 2 seconds.')
        else:
            pass
    except KeyError as e:
        print('\nError retrieving X-RateLimit for token %d: %s\n' % (token_index, e.args))


def get_access_tokens():
    """
    Retrieves GitHub Search API authentication tokens from files.

    :return: List of token lists
    """
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


def extract_from_readme(response):
    """
    Gets plain text from readme file.

    :param response: Res
    :return: Plain plain_text of readme file
    """
    plain_text = None
    link_list = []
    reference_list = []
    # Request successful
    if response.status_code == 200:

        # Extract plain_text and remove all line breaks
        soup = BeautifulSoup(response.text, features='lxml')

        # Find all arxiv links and append to reference_list
        for reference in soup.findAll('a', attrs={'href': re.compile('(arxiv|ieee.org)')}):
            reference_list.append(reference.get('href'))

        # Find all links and append to link_list
        for link in soup.findAll('a', attrs={'href': re.compile('^http://')}):
            link_list.append(link.get('href'))

        # Remove references from link_list
        link_list = list(set(link_list) - set(reference_list))

        plain_text = ''.join(soup.find_all(text=True))
        plain_text = plain_text.replace('\n', ' ').replace('\t', ' ').replace('\r', '')

        # Set plain_text to null if empty string
        if plain_text == '':
            plain_text = None

    # Request unsuccessful
    elif response.status_code == 404:
        pass
        # print(' - Repository without readme found for: %s\n%s' % (response.plain_text, response.request.url))

    else:
        print('Unknown error occurred while parsing readme file: %d' % response.status_code)
        print(response.reason)
        if response.status_code == 403:
            print('Access denied.')

    # print('\n\nLimit: %d' % int(response.headers['X-RateLimit-Limit']))
    # print('Remaining: %d' % int(response.headers['X-RateLimit-Remaining']))

    return plain_text, link_list, reference_list