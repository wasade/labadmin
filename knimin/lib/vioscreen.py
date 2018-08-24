import requests
import os
import functools
import json
import pandas as pd

from datetime import datetime
from knimin import config
from knimin.lib.data_access import SQLHandler

class VioscreenHandler(object):
    def __init__(self):
        self._session = requests.Session()
        # setup our HTTP header data
        self._headers = {'Accept': 'application/json',
                         'Authorization': 'Bearer %s' % self._get_token()}
        self._users = self.get_users()
        self.sql_handler = SQLHandler(config)

    # get an API token
    def _get_token(self):
        return self._post('https://api.viocare.com/KLUCB/auth/login',
                     data={"username": "APIAdminKLUCB", 
                           "password": "APIAdminKLUCB"})['token']
    
    def get_users(self):
        return self._get('https://api.viocare.com/KLUCB/users',
                         headers=self._headers)

    def _get(self, url, retries=5, **kwargs):
        for i in range(retries):
            req = self._session.get(url, **kwargs)
            if req.status_code != 200:  # HTTP status code, 200 is all good
                data = req.json()

                # if we did not get a HTTP status code 200, than guess that the
                # API token is no longer valid so get a new one and retry
                if 'Code' in data and data['Code'] == 1016:
                    self._headers['token'] = self._get_token()
                else:
                    print(self._session.get, url, kwargs)
                    raise ValueError("Unable to make this query work")
            else:
                return req.json()
        raise ValueError("Unable to make this query work")

    def _post(self, url, retries=5, **kwargs):
        for i in range(retries):
            req = self._session.post(url, **kwargs)
            if req.status_code != 200:  # HTTP status code, 200 is all good
                data = req.json()

                # if we did not get a HTTP status code 200, than guess that the
                # API token is no longer valid so get a new one and retry
                if 'Code' in data and data['Code'] == 1016:
                    self._headers['token'] = self._get_token()
                else:
                    print(self._session.post, url, kwargs)
                    raise ValueError("Unable to make this query work")
            else:
                return req.json()
        raise ValueError("Unable to make this query work")

    # restructure the data so that the "survey_id" is
    # associated with each row of data
    def tidyfy(self, username, payload):
        dat = []
        for entry in payload:
            entry['survey_id'] = username
            dat.append(entry)
        return dat

    def get_session_data(self, session_id, endpoint):
        return self._get('https://api.viocare.com/KLUCB/sessions/%s/%s' %
                         (session_id, endpoint),
                         headers=self._headers)

    def get_vioscreen(self, limit=None):
        # assign local users var to be derivative of users member variable
        try:
            user_ids = {x['username'] for x in self._users['users'][:limit]}
        except TypeError:
            raise TypeError('limit should be type int')

        # takes all survey IDs from vio_screen survey info and filters
        # only ones that do not have their data in the ag database
        ids_to_sync = self.get_vio_survey_ids_not_in_ag(user_ids)

        # gets all survey info of ids_to_sync and updates users with filtered surveys
        users = {}
        users_to_sync = []
        for i in self._users['users']:
            if i['username'] in ids_to_sync:
                users_to_sync.append(i)
        users['users'] = users_to_sync

        #tidy_data = []
        have_results_for = set()

        #if os.path.exists('vioscreen_tmp_out.txt'):
        #    tidy_data = [json.loads(l) for l in open('vioscreen_tmp_out.txt')]
        #    have_results_for = {d['survey_id'] for d in tidy_data}
        #else:
        #    print('Creating vioscreen_tmp_out.txt')

        #tmp_out = open('vioscreen_tmp_out.txt', 'a')
 
        for idx, user in enumerate(users['users']):
            username = user['username']
            if username in have_results_for:
                continue

            try:
                session_data = self._get('https://api.viocare.com/KLUCB/users/%s/sessions'
                                         % username, headers=self._headers)
            except ValueError:
                # I don't understand this, but "JDebelius" does not exist.
                # must have been a test account since it's not an AG survey id
                continue

            for session_detail in session_data['sessions']:
                session_id = session_detail['sessionId']
                detail = self._get('https://api.viocare.com/KLUCB/sessions/%s/detail'
                                   % session_id, headers=self._headers)

                # only finished surveys will have their data pulled
                if detail['status'] != 'Finished':
                    print(username, 'NOT FINISHED')
                    continue

                # only get the first finished one, not sure how to handle a situation if someone has multiple right now
                try:
                    foodcomponents = self.get_session_data(session_id, 'foodcomponents')['data']
                    percentenergy = self.get_session_data(session_id, 'percentenergy')['calculations']
                    mpeds = self.get_session_data(session_id, 'mpeds')['data']
                    eatingpatterns = self.get_session_data(session_id, 'eatingpatterns')['data']
                    foodconsumption = self.get_session_data(session_id, 'foodconsumption')['foodConsumption']
                    dietaryscore = self.get_session_data(session_id, 'dietaryscore')['dietaryScore']['scores']
                except ValueError:
                    # sometimes there is a status Finished w/o data...
                    continue

                foodcomponents = self.tidyfy(username, foodcomponents)
                percentenergy = self.tidyfy(username, percentenergy)
                mpeds = self.tidyfy(username, mpeds)
                eatingpatterns = self.tidyfy(username, eatingpatterns)
                foodconsumption = self.tidyfy(username, foodconsumption)
                dietaryscore = self.tidyfy(username, dietaryscore)

                self.insert_foodcomponents(foodcomponents)
                self.insert_percentenergy(percentenergy)
                self.insert_mpeds(mpeds)
                self.insert_eatingpatterns(mpeds)
                self.insert_foodconsumption(foodconsumption)
                self.insert_dietaryscore(dietaryscore)

                #tidy_data.extend(a)
                #tidy_data.extend(b)
                #tidy_data.extend(c)
                #tidy_data.extend(d)
                #tidy_data.extend(e)
                #tidy_data.extend(f)

                #for item in a + b + c + d + e + f:
                #    tmp_out.write(json.dumps(item))
                #    tmp_out.write('\n')

                have_results_for.add(username)
                break

            # prints time for every ten surveys finished
            if idx % 10 == 0:
                print(datetime.now(), idx)

        #df = pd.DataFrame(tidy_data)
        #df.to_csv('vioscreen_dump.tsv', sep='\t', index=False)

        #in_file = open('vioscreen_dump.tsv', 'rw+')
        #db.store_external_survey(in_file, 'Vioscreen')

    # DB access functions
#    def update_vio_status(self, survey_id, status):
#        sql = """SELECT survey_id from ag.vioscreen_surveys"""
#        survey_ids = 
#
#    def get_vio_survey_ids_in_ag(self):
#        sql = """SELECT survey_id FROM ag.vioscreen_surveys"""
#        return self.sql_handler.execute_fetchall(sql)
#
    def get_vio_survey_ids_not_in_ag(self, vio_ids):
        """Retrieve survey ids that have vioscreen data but
           have not have their data transferred to ag

        Parameters
        ----------
        vio_ids : set of ids present in vioscreen

        Returns
        -------
        set of str
            The set of survey_ids in vioscreen that aren't in ag
        """
        sql = """SELECT survey_id FROM ag.external_survey_answers"""

        ag_survey_ids = self.sql_handler.execute_fetchall(sql)
        ag_survey_ids = {i[0] for i in ag_survey_ids}
        return vio_ids - set(ag_survey_ids)

    def _call_sql_handler(self, sql, session_data):
        """Formats session_data to insert into a particular table

        Parameters
        ----------
        sql : SQL query specific to particular session insertion
        session_data : Data pulled from Vioscreen

        Return
        ------
        int
            The number of rows added to the database
        """
        inserts = []
        for row in session_data:
            inserts.append([row[key] for key in row])
        self.sql_handler.executemany(sql, inserts)
        return len(inserts)

    def insert_foodcomponents(self, foodcomponents):
        """Inserts foodcomponents data into AG database

        Parameters
        ----------
        foodcomponents : foodcomponents session data

        Return
        ------
        int
            The number of rows added to the database
        """
        sql = """INSERT INTO ag.vioscreen_foodcomponents (code, description,
                 valueType, amount, units, survey_id) VALUES (%s,
                 %s, %s, %s, %s, %s)"""
        return self._call_sql_handler(sql, foodcomponents)

    def insert_percentenergy(self, percentenergy):
        """Inserts percentenergy data into AG database

        Parameters
        ----------
        percentenergy : percentenergy session data

        Return
        ------
        int
            The number of rows added to the database
        """
        sql = """INSERT INTO ag.vioscreen_percentenergy (code, description,
                 precision, foodComponentType, amount, foodDataDefinition,
                 units, survey_id, shortDescription) VALUES (%s, %s, %s, %s,
                 %s, %s, %s, %s, %s)"""
        return self._call_sql_handler(sql, percentenergy)

    def insert_mpeds(self, mpeds):
        """Inserts mpeds data into AG database

        Parameters
        ----------
        mpeds : mpeds session data

        Return
        ------
        int
            The number of rows added to the database
        """
        sql = """INSERT INTO ag.vioscreen_mpeds (code, description, valueType,
                 amount, units, survey_id) VALUES (%s, %s, %s, %s, %s, %s)"""
        return self._call_sql_handler(sql, mpeds)

    def insert_eatingpatterns(self, eatingpatterns):
        """Inserts eatingpatterns data into AG database

        Parameters
        ----------
        eatingpatterns : eatingpatterns session data

        Return
        ------
        int
            The number of rows added to the database
        """
        sql = """INSERT INTO ag.vioscreen_eatingpatterns (code,
                 description, valueType, amount, units, survey_id)
                 VALUES (%s, %s, %s, %s, %s, %s)"""
        return self._call_sql_handler(sql, eatingpatterns)

    def insert_foodconsumption(self, foodconsumption):
        """Inserts foodconsumption data into AG database

        Parameters
        ----------
        foodconsumption : foodconsumption session data

        Return
        ------
        int
            The number of rows added to the database
        """
        sql = """INSERT INTO ag.vioscreen_foodconsumption
                 (consumptionAdjustment, description, created,
                 servingFrequencyText, amount, frequency, foodGroup,
                 servingSizeText, foodCode, survey_id, data) VALUES 
                 (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        # convert large data dict to json for data storage
        for row in foodconsumption:
            row['data'] = json.dumps(row['data'])
        return self._call_sql_handler(sql, foodconsumption) 

    def insert_dietaryscore(self, dietaryscore):
        """Inserts dietaryscore data into AG database

        Parameters
        ----------
        dietaryscore : dietaryscore session data

        Return
        ------
        int
            The number of rows added to the database
        """
        sql = """INSERT INTO ag.vioscreen_dietaryscore (name, lowerLImit,
                 score, survey_id, type, upperLimit) VALUES (%s, %s,
                 %s, %s, %s, %s)"""
        return self._call_sql_handler(sql, dietaryscore)
