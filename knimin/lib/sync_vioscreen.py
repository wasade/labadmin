import json
import requests
import functools
from datetime import datetime

session = requests.Session()

# get an API token
def get_token():
    return post('https://api.viocare.com/KLUCB/auth/login',
                 data={ "username": "APIAdminKLUCB", "password": "APIAdminKLUCB"})['token']

# issue a request and get some data back, allow for retrying in case a query fails
def make_request(method, url, retries=5, **kwargs):
    global headers
    for i in range(retries):
        req = method(url, **kwargs)
        if req.status_code != 200:  # HTTP status code, 200 is all good
            data = req.json()

            # if we did not get a HTTP status code 200, than guess that the
            # API token is no longer valid so get a new one and allow a retry
            if 'Code' in data and data['Code'] == 1016:
                headers['token'] = get_token()
            else:
                print(method, url, kwargs)
                raise ValueError("Unable to make this query work")
        else:
            return req.json()
    raise ValueError("Unable to make this query work")

# make a get and post function
get = functools.partial(make_request, session.get)
post = functools.partial(make_request, session.post)

# setup our HTTP header data
headers = {'Accept': 'application/json',
           'Authorization': 'Bearer %s' % get_token()}

# these are all of the survey IDs
users = get('https://api.viocare.com/KLUCB/users', headers=headers)

user_ids = {x['username'] for x in users['users']}
