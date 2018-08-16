import json
import requests
import functools
from datetime import datetime
from knimin import db

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

# restructure the data so that the "survey_id" is associated with
# each row of data
def tidyfy(username, payload):
    dat = []
    for entry in payload:
        entry['survey_id'] = username
        dat.append(entry)
    return dat

# make a get and post function
get = functools.partial(make_request, session.get)
post = functools.partial(make_request, session.post)

# setup our HTTP header data
headers = {'Accept': 'application/json',
           'Authorization': 'Bearer %s' % get_token()}

# get session specific data, this is different components of the data that vioscreen produces
def make_get_session_data(endpoint, headers, session_id):
    return get('https://api.viocare.com/KLUCB/sessions/%s/%s' % (session_id,
                                                                 endpoint),
               headers=headers)

get_foodcomponents = functools.partial(make_get_session_data, 'foodcomponents', headers)
get_percentenergy = functools.partial(make_get_session_data, 'percentenergy', headers)
get_mpeds = functools.partial(make_get_session_data, 'mpeds', headers)
get_eatingpatterns = functools.partial(make_get_session_data, 'eatingpatterns', headers)
get_foodconsumption = functools.partial(make_get_session_data, 'foodconsumption', headers)
get_dietaryscore = functools.partial(make_get_session_data, 'dietaryscore', headers)

# these are all of the survey IDs
users = get('https://api.viocare.com/KLUCB/users', headers=headers)

# takes all survey IDs from vio_screen survey info and filters
# only ones that do not have their data in the ag database
user_ids = {x['username'] for x in users['users']}
ids_to_sync = db.get_vio_survey_ids_not_in_ag(user_ids)

# gets all survey info of ids_to_sync and updates users with filtered surveys
users_to_sync = []
for i in users['users']:
    if i['username'] in ids_to_sync:
        users_to_sync.append(i)
users['users'] = users_to_sync[:5]

not_complete = []
complete = []
tidy_data = []
have_results_for = set()

import os
if os.path.exists('vioscreen_tmp_out.txt'):
    tidy_data = [json.loads(l) for l in open('vioscreen_tmp_out.txt')]
    have_results_for = {d['survey_id'] for d in tidy_data}

else:
    print('Creating vioscreen_tmp_out.txt')

tmp_out = open('vioscreen_tmp_out.txt', 'a')

for idx, user in enumerate(users['users']):
    username = user['username']
    print(username)
    if username in have_results_for:
        print('username in have_results_for')
        continue

    try:
        session_data = get('https://api.viocare.com/KLUCB/users/%s/sessions' % username, headers=headers)
    except ValueError:
        # I don't understand this, but "JDebelius" does not exist.
        # must have been a test account since it's not an AG survey id
        continue

    any_finished = False
    for session_detail in session_data['sessions']:
        session_id = session_detail['sessionId']
        detail = get('https://api.viocare.com/KLUCB/sessions/%s/detail' % session_id, headers=headers)

        # only finished surveys will have their data pulled
        if detail['status'] != 'Finished':
            continue
        else:
            any_finished = True

        # only get the first finished one, not sure how to handle a situation if someone has multiple right now
        try:
            foodcomponents = get_foodcomponents(session_id)['data']
            percentenergy = get_percentenergy(session_id)['calculations']
            mpeds = get_mpeds(session_id)['data']
            eatingpatterns = get_eatingpatterns(session_id)['data']
            foodconsumption = get_foodconsumption(session_id)['foodConsumption']
            dietaryscore = get_dietaryscore(session_id)['dietaryScore']['scores']
        except ValueError:
            # sometimes there is a status Finished w/o data...
            continue

        a = tidyfy(username, foodcomponents)
        b = tidyfy(username, percentenergy)
        c = tidyfy(username, mpeds)
        d = tidyfy(username, eatingpatterns)
        e = tidyfy(username, foodconsumption)
        f = tidyfy(username, dietaryscore)

        tidy_data.extend(a)
        tidy_data.extend(b)
        tidy_data.extend(c)
        tidy_data.extend(d)
        tidy_data.extend(e)
        tidy_data.extend(f)

        for item in a + b + c + d + e + f:
            tmp_out.write(json.dumps(item))
            tmp_out.write('\n')

        have_results_for.add(username)
        break

    if not any_finished:
        not_complete.append(username)


    if idx % 10 == 0:
        print(datetime.now(), idx)

import pandas as pd
df = pd.DataFrame(tidy_data)
df.to_csv('vioscreen_dump.tsv', sep='\t', index=False)
