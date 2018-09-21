import requests
import os
import functools
import json
import pandas as pd

from datetime import datetime
from knimin import config
from knimin.lib.data_access import SQLHandler

class VioscreenHandler(object):
    """VioScreen handler object. Used to pull data from VioScreen
       RESTful API and store data in AG database.
    """
    def __init__(self):
        with open('./auth.json') as f:
            self._key = json.load(f)['key']
            self._user = json.load(f)['user']
            self._pw = json.load(f)['pw']
        self._session = requests.Session()
        # setup our HTTP header data
        self._headers = {'Accept': 'application/json',
                         'Authorization': 'Bearer %s' % self.get_token()}
        self._users = self.get_users()
        self.sql_handler = SQLHandler(config)

    # get an API token
    def get_token(self):
        """Gets an API token for vioscreen

        Return
        ------
        str
            The API token
        """
        return self.post('https://api.viocare.com/%s/auth/login' % self._key,
                         data={"username": self._user, 
                               "password": self._pw})['token']
    
    def get_users(self):
        """Gets list of users that vioscreen has data for

        Return
        ------
        list of dict
            List of users that have vioscreen data
        """
        return self.get('https://api.viocare.com/%s/users' % self._key,
                         headers=self._headers)

    def get(self, url, retries=5, **kwargs):
        """Extension of get method from requests. Will get the requested
           data and return it, or return an error message if data was
           unable to be retrieved

           Parameters
           ----------
           url
               The url from which data is requested
           retries
               Number of tries the function takes if we refresh the token
               each time a retrieval fails
           **kwargs
               Optional arguments that requests takes
        
           Return
           ------
           dict
               Data returned from get request
        """
        for i in range(retries):
            req = self._session.get(url, **kwargs)
            if req.status_code != 200:  # HTTP status code, 200 is all good
                data = req.json()

                # if we did not get a HTTP status code 200, than guess that the
                # API token is no longer valid so get a new one and retry
                if 'Code' in data and data['Code'] == 1016:
                    self._headers['token'] = self.get_token()
                else:
                    raise ValueError("Unable to make this query work")
            else:
                return req.json()
        raise ValueError("Unable to make this query work")

    def post(self, url, retries=5, **kwargs):
        """Extension of post method from requests. Will post and return
           requested data and return it, or return an error message if data was
           unable to be retrieved

           Parameters
           ----------
           url
               The url from which data is requested
           retries
               Number of tries the function takes if we refresh the token
               each time a retrieval fails
           **kwargs
               Optional arguments that requests takes
           
           Return
           ------
           dict
               Data returned from post request
        """
        for i in range(retries):
            req = self._session.post(url, **kwargs)
            if req.status_code != 200:  # HTTP status code, 200 is all good
                data = req.json()

                # if we did not get a HTTP status code 200, than guess that the
                # API token is no longer valid so get a new one and retry
                if 'Code' in data and data['Code'] == 1016:
                    self._headers['token'] = self.get_token()
                else:
                    raise ValueError("Unable to make this query work")
            else:
                return req.json()
        raise ValueError("Unable to make this query work")

    def tidyfy(self, username, payload):
        """Restructures data so that 'survey_id' is associated with each row

        Parameters
        ----------
        username: str
            Survey ID that is being inserted each of the rows
        payload: list of dict
            The data that is getting associated with the survey_id

        Return
        ------
        list of dict
            The newly formatted data containing the username in each row
        """
        dat = []
        for entry in payload:
            entry['survey_id'] = username
            dat.append(entry)
        return dat

    def get_session_data(self, session_id, endpoint):
        """Pulls data from the vioscreen API based on 
           a specific session ID and session type(ex. 'foodcomponents')

        Parameters
        ----------
        session_id: str
            Session ID that data is being requested for
        endpoint: str
            Name of the session type that data is being requested for

        Return
        ------
        dict:
            Food frequency questionnaire data
        """
        return self.get('https://api.viocare.com/%s/sessions/%s/%s' %
                         (self._key, session_id, endpoint),
                         headers=self._headers)

    def sync_vioscreen(self, user_ids=None):
        """Pulls data from the vioscreen API and stores
           the data into the AG database

        Parameters
        ----------
        user_ids: set of str
            Set of user_ids (identical to survey_ids) that
            are needed to have their data pulled. Default None (syncs all)
        """
        if user_ids:
            if type(user_ids) is not set:
                raise TypeError('user_ids should be type set')
        else:
            user_ids = {x['username'] for x in self._users['users']}

        # takes all survey IDs from vio_screen survey info and filters
        # only ones that do not have their data in the ag database
        ids_to_sync = self.get_vio_survey_ids_not_in_ag(user_ids)

        # gets all survey info of ids_to_sync and updates users
        users = {'users': []}
        for user in self._users['users']:
            if user['username'] in ids_to_sync:
                users['users'].append(user)

        # gets list of surveys in AG database along with their statuses
        survey_ids = self.get_init_surveys()

        for user in users['users']:
            username = user['username']

            try:
                session_data = self.get('https://api.viocare.com/%s/users/%s/sessions'
                                        % (self._key, username), headers=self._headers)
            except ValueError:
                # I don't understand this, but "JDebelius" does not exist.
                # must have been a test account since it's not an AG survey id
                continue

            for session_detail in session_data['sessions']:
                session_id = session_detail['sessionId']
                detail = self.get('https://api.viocare.com/%s/sessions/%s/detail'
                                  % (self._key, session_id), headers=self._headers)

                # Adds new survey information to database
                if username not in survey_ids:
                    survey_ids[username] = detail['status']
                    self.insert_survey(username, detail['status'])
                # Updates status of vioscreen survey if it has changed
                elif survey_ids[username] != detail['status']:
                    survey_ids[username] = detail['status']
                    self.update_status(username, detail['status'])

                # only finished surveys will have their data pulled
                if detail['status'] != 'Finished':
                    continue

                try:
                    foodcomponents = self.get_session_data(session_id,\
                        'foodcomponents')['data']
                    percentenergy = self.get_session_data(session_id,\
                        'percentenergy')['calculations']
                    mpeds = self.get_session_data(session_id, 'mpeds')['data']
                    eatingpatterns = self.get_session_data(session_id,\
                        'eatingpatterns')['data']
                    foodconsumption = self.get_session_data(session_id,\
                        'foodconsumption')['foodConsumption']
                    dietaryscore = self.get_session_data(session_id,\
                        'dietaryscore')['dietaryScore']['scores']
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
                self.insert_eatingpatterns(eatingpatterns)
                self.insert_foodconsumption(foodconsumption)
                self.insert_dietaryscore(dietaryscore)

                break

    # DB access functions
    def get_init_surveys(self):
        """Retrieve initial set of vioscreen surveys before sync

        Returns
        -------
        dict
           Initial set of survey IDs and their corresponding statuses 
        """
        sql = """SELECT survey_id, status from ag.vioscreen_surveys"""
        data = self.sql_handler.execute_fetchall(sql)
        survey_ids = {}
        for row in data:
            survey_ids[row[0]] = row[1]
        return survey_ids

    def update_status(self, survey_id, status):
        """Updates vioscreen status of AG database to correspond to status
           pulled from vioscreen

        Parameters
        ----------
        survey_id: str
            Survey ID being updated in database
        status: str
            Status that the survey ID status is being updated to
        """
        if status == 'Finished':
            pulldown_date = datetime.now()
        else:
            pulldown_date = None
        sql = """UPDATE ag.vioscreen_surveys SET status=%s,
                 pulldown_date=%s WHERE survey_id=%s"""
        self.sql_handler.execute(sql, [status, pulldown_date, survey_id])

    def insert_survey(self, survey_id, status):
        """Inserts a survey id that has a vioscreen session along with its
           status ('Started', 'Finished', etc.) and pulldown date into the
           ag.vioscreen_surveys table

        Parameters
        ----------
        survey_id: str
            Survey ID being inserted into vioscreen survey database    
        status: str
            Status that the survey ID is being inserted with
        """
        pulldown_date = datetime.now()
        sql = """INSERT INTO ag.vioscreen_surveys (status, survey_id,
                 pulldown_date) VALUES (%s, %s, %s)"""
        self.sql_handler.execute(sql, [status, survey_id, pulldown_date])

    def get_vio_survey_ids_not_in_ag(self, vio_ids):
        """Retrieve survey ids that have vioscreen data but
           have not have their data transferred to AG

        Parameters
        ----------
        vio_ids: set of str
            The set of IDs present in vioscreen

        Returns
        -------
        set of str
            The set of survey IDs in vioscreen that aren't in AG
        """
        sql = """SELECT survey_id FROM ag.vioscreen_surveys
                 WHERE status = 'Finished'"""

        ag_survey_ids = self.sql_handler.execute_fetchall(sql)
        ag_survey_ids = {i[0] for i in ag_survey_ids}
        return vio_ids - set(ag_survey_ids)

    def _call_sql_handler(self, sql, session_data):
        """Formats session_data to insert into a particular table

        Parameters
        ----------
        sql: str
            SQL query specific to particular session insertion
            session_data : Data pulled from Vioscreen

        Return
        ------
        int
            The number of rows added to the database
        """
        # inserts represents the data of a session to be stored
        inserts = []
        keys = sorted(session_data[0].keys())
        for row in session_data:
            # row_insert represents the data of a single row
            row_insert = []
            for key in keys:
                row_insert.append(row[key])
            inserts.append(row_insert)
        self.sql_handler.executemany(sql, inserts)
        return len(inserts)

    def insert_foodcomponents(self, foodcomponents):
        """Inserts foodcomponents data into AG database

        Parameters
        ----------
        foodcomponents: list of different types
            foodcomponents session data

        Return
        ------
        int
            The number of rows added to the database
        """
        sql = """INSERT INTO ag.vioscreen_foodcomponents (amount, code,
                 description, survey_id, units, valueType) VALUES (%s,
                 %s, %s, %s, %s, %s)"""
        return self._call_sql_handler(sql, foodcomponents)

    def insert_percentenergy(self, percentenergy):
        """Inserts percentenergy data into AG database

        Parameters
        ----------
        percentenergy: list of different types
            percentenergy session data

        Return
        ------
        int
            The number of rows added to the database
        """
        sql = """INSERT INTO ag.vioscreen_percentenergy
                    (amount, code, description, foodComponentType,
                     foodDataDefinition, precision, shortDescription,
                     survey_id, units)
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        return self._call_sql_handler(sql, percentenergy)

    def insert_mpeds(self, mpeds):
        """Inserts mpeds data into AG database

        Parameters
        ----------
        mpeds: list of different types
            mpeds session data

        Return
        ------
        int
            The number of rows added to the database
        """
        sql = """INSERT INTO ag.vioscreen_mpeds
                    (amount, code, description, survey_id,
                     units, valueType)
                 VALUES (%s, %s, %s, %s, %s, %s)"""
        return self._call_sql_handler(sql, mpeds)

    def insert_eatingpatterns(self, eatingpatterns):
        """Inserts eatingpatterns data into AG database

        Parameters
        ----------
        eatingpatterns: list of different types
            eatingpatterns session data

        Return
        ------
        int
            The number of rows added to the database
        """
        sql = """INSERT INTO ag.vioscreen_eatingpatterns
                    (amount, code, description,
                     survey_id, units, valueType)
                 VALUES (%s, %s, %s, %s, %s, %s)"""
        return self._call_sql_handler(sql, eatingpatterns)

    def insert_foodconsumption(self, foodconsumption):
        """Inserts foodconsumption data into AG database

        Parameters
        ----------
        foodconsumption: list of different types
            foodconsumption session data

        Return
        ------
        int
            The number of rows added to the database
        """
        sql = """INSERT INTO ag.vioscreen_foodconsumption
                    (amount, consumptionAdjustment, created, data, description,
                     foodCode, foodGroup, frequency, servingFrequencyText,
                     servingSizeText, survey_id)
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        # convert large data dict to json for data storage
        for row in foodconsumption:
            row['data'] = json.dumps(row['data'])
        return self._call_sql_handler(sql, foodconsumption) 

    def insert_dietaryscore(self, dietaryscore):
        """Inserts dietaryscore data into AG database

        Parameters
        ----------
        dietaryscore: list of different types
            dietaryscore session data

        Return
        ------
        int
            The number of rows added to the database
        """
        sql = """INSERT INTO ag.vioscreen_dietaryscore 
                    (lowerLimit, name, score, survey_id, 
                     type, upperLimit) 
                 VALUES (%s, %s, %s, %s, %s, %s)"""
        return self._call_sql_handler(sql, dietaryscore)

    # Testing function
    def flush_vioscreen_db(self):
        """Flushes VioScreen data from AG database"""
        tables = ['foodcomponents', 'percentenergy', 'mpeds', 'eatingpatterns',
                  'foodconsumption', 'dietaryscore', 'surveys']
        for i in tables:
            sql = """DELETE FROM ag.vioscreen_{0}""".format(i)
            self.sql_handler.execute(sql)
