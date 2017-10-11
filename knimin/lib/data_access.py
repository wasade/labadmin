from __future__ import unicode_literals
from contextlib import contextmanager
from collections import defaultdict, namedtuple
from os import walk
from os.path import join, splitext, isdir, abspath
from copy import copy
from re import sub
from hashlib import sha512
from datetime import datetime, time, timedelta
from requests.exceptions import SSLError
import json
import re
import pandas as pd
import numpy as np

from bcrypt import hashpw, gensalt
from future.utils import viewitems

from psycopg2 import connect, Error as PostgresError
from psycopg2.extras import DictCursor

from mail import send_email
from util import (make_valid_kit_ids, make_verification_code, make_passwd,
                  categorize_age, categorize_etoh, categorize_bmi, correct_age,
                  fetch_url, correct_bmi)
from tornado.escape import xhtml_escape
from constants import (md_lookup, month_int_lookup, month_str_lookup,
                       regions_by_state, blanks_values, season_lookup,
                       ebi_remove, env_lookup)
from geocoder import geocode, Location, GoogleAPILimitExceeded
from string_converter import converter


DATAFRAME_SURVEY_COLUMNS = ('survey',
                            'participant_survey_id',
                            'barcode',
                            'question',
                            'answer')


class IncorrectEmailError(Exception):
    pass


class IncorrectPasswordError(Exception):
    pass


class SQLHandler(object):
    """Encapsulates the DB connection with the Postgres DB

    Sourced from QIITA's SQLConnectionHandler
    """
    def __init__(self, config):
        self._connection = connect(user=config.db_user,
                                   password=config.db_password,
                                   database=config.db_database,
                                   host=config.db_host,
                                   port=config.db_port)

    def __del__(self):
        self._connection.close()

    @contextmanager
    def cursor(self):
        """ Returns a Postgres cursor

        Returns
        -------
        pgcursor : psycopg2.cursor
        """
        with self._connection.cursor(cursor_factory=DictCursor) as cur:
            yield cur

    def _check_sql_args(self, sql_args):
        """ Checks that sql_args have the correct type

        Inputs:
            sql_args: SQL arguments

        Raises a TypeError if sql_args does not have the correct type,
            otherwise it just returns the execution to the caller
        """
        # Check that sql arguments have the correct type
        if sql_args and type(sql_args) not in [tuple, list, dict]:
            raise TypeError("sql_args should be tuple, list or dict. Found %s "
                            % type(sql_args))

    @contextmanager
    def _sql_executor(self, sql, sql_args=None, many=False):
        """Executes an SQL query

        Parameters
        ----------
        sql: str
            The SQL query
        sql_args: tuple or list, optional
            The arguments for the SQL query
        many: bool, optional
            If true, performs an execute many call

        Returns
        -------
        pgcursor : psycopg2.cursor
            The cursor in which the SQL query was executed

        Raises
        ------
        ValueError
            If there is some error executing the SQL query
        """
        # Check that sql arguments have the correct type
        if many:
            for args in sql_args:
                self._check_sql_args(args)
        else:
            self._check_sql_args(sql_args)

        # Execute the query
        with self.cursor() as cur:
            try:
                if many:
                    cur.executemany(sql, sql_args)
                else:
                    cur.execute(sql, sql_args)
                yield cur
                self._connection.commit()
            except PostgresError as e:
                self._connection.rollback()
                try:
                    err_sql = cur.mogrify(sql, sql_args)
                except:
                    err_sql = cur.mogrify(sql, sql_args[0])
                # errors might contain user strings encoded in utf-8.
                raise ValueError(("\nError running SQL query: %s"
                                  "\nError: %s" % (err_sql.decode('utf-8'),
                                                   str(e).decode('utf-8'))))

    def execute_fetchall(self, sql, sql_args=None):
        """ Executes a fetchall SQL query

        Parameters
        ----------
        sql: str
            The SQL query
        sql_args: tuple or list, optional
            The arguments for the SQL query

        Returns
        ------
        list of tuples
            The results of the fetchall query

        Note: from psycopg2 documentation, only variable values should be bound
            via sql_args, it shouldn't be used to set table or field names. For
            those elements, ordinary string formatting should be used before
            running execute.
        """
        with self._sql_executor(sql, sql_args) as pgcursor:
            result = pgcursor.fetchall()
        return result

    def execute_fetchone(self, sql, sql_args=None):
        """ Executes a fetchone SQL query

        Parameters
        ----------
        sql: str
            The SQL query
        sql_args: tuple or list, optional
            The arguments for the SQL query

        Returns
        -------
        Tuple
            The results of the fetchone query

        Notes
        -----
        from psycopg2 documentation, only variable values should be bound
            via sql_args, it shouldn't be used to set table or field names. For
            those elements, ordinary string formatting should be used before
            running execute.
        """
        with self._sql_executor(sql, sql_args) as pgcursor:
            result = pgcursor.fetchone()
        return result

    def execute_fetchdict(self, sql, sql_args=None):
        """ Executes a fetchall SQL query and returns each row as a dict

        Parameters
        ----------
        sql: str
            The SQL query
        sql_args: tuple or list, optional
            The arguments for the SQL query

        Returns
        -------
        list of dict
            The results of the query as
            [{colname: val, colname: val, ...}, ...]

        Notes
        -----
        from psycopg2 documentation, only variable values should be bound
        via sql_args, it shouldn't be used to set table or field names.
        For those elements, ordinary string formatting should be used
        before running execute.
        """
        with self._sql_executor(sql, sql_args) as pgcursor:
            result = [dict(row) for row in pgcursor.fetchall()]
        return result

    def execute(self, sql, sql_args=None):
        """ Executes an SQL query with no results

        Parameters
        ----------
        sql: str
            The SQL query
        sql_args: tuple or list, optional
            The arguments for the SQL query

        Notes
        -----
        from psycopg2 documentation, only variable values should be bound
            via sql_args, it shouldn't be used to set table or field names. For
            those elements, ordinary string formatting should be used before
            running execute.
        """
        with self._sql_executor(sql, sql_args):
            pass

    def executemany(self, sql, sql_args_list):
        """ Executes an executemany SQL query with no results

        Parameters
        ----------
        sql: str
            The SQL query
        sql_args: list of tuples
            The arguments for the SQL query

        Note: from psycopg2 documentation, only variable values should be bound
            via sql_args, it shouldn't be used to set table or field names. For
            those elements, ordinary string formatting should be used before
            running execute.
        """
        with self._sql_executor(sql, sql_args_list, True):
            pass

    def execute_proc_return_cursor(self, procname, proc_args):
        """Executes a stored procedure and returns a cursor

        Parameters
        ----------
        procname: str
            the name of the stored procedure
        proc_args: list
            arguments sent to the stored procedure
        """
        proc_args.append('cur2')
        cur = self._connection.cursor()
        cur.callproc(procname, proc_args)
        cur.close()
        return self._connection.cursor('cur2')


class KniminAccess(object):
    # arbitrary, unique ID and value
    human_sites = ['Stool',
                   'Mouth',
                   'Right hand',
                   'Left hand',
                   'Forehead',
                   'Torso',
                   'Left leg',
                   'Right leg',
                   'Nares',
                   'Hair',
                   'Tears',
                   'Nasal mucus',
                   'Ear wax',
                   'Vaginal mucus']

    animal_sites = ['Stool',
                    'Mouth',
                    'Nares',
                    'Ears',
                    'Skin',
                    'Fur']

    general_sites = ['Animal Habitat',
                     'Biofilm',
                     'Dust',
                     'Food',
                     'Fermented Food',
                     'Indoor Surface',
                     'Outdoor Surface',
                     'Plant habitat',
                     'Soil',
                     'Sole of shoe',
                     'Water']

    def __init__(self, config):
        self._con = SQLHandler(config)
        self._con.execute('set search_path to ag, barcodes, public')
        self.config = config

    def _get_col_names_from_cursor(self, cur):
        if cur.description:
            return [x[0] for x in cur.description]
        else:
            return []

    def has_access(self, email, access_levels):
        """Whether user has access level given or not.

        Parameters
        ----------
        email : str
            Email of user to check
        access_levels : list of str
            Access level to check

        Returns
        -------
        bool
            Whether user has access (true) or not (false)

        Notes
        -----
        For uses with Admin acces, this will always return true.

        Raises
        ------
        ValueError
            Unknown access level passed
        """
        # Make sure all access levels passed exist
        sql = "Select 1 from ag.labadmin_access WHERE access_name = %s"
        for level in access_levels:
            if self._con.execute_fetchone(sql, [level]) is None:
                raise ValueError('Unknown access level %s' % level)

        sql = """SELECT EXISTS(
                    SELECT 1
                    FROM ag.labadmin_users_access
                    JOIN ag.labadmin_access USING (access_id)
                    WHERE email = %s AND access_name IN %s)"""
        access = tuple(access_levels + ['Admin'])
        return self._con.execute_fetchone(sql, [email, access])[0]

    def get_users(self):
        """Get a list of users in the system

        Returns
        -------
        list of str
            Users in the system
        """
        sql = "SELECT email FROM ag.labadmin_users"
        hold = self._con.execute_fetchall(sql)
        if hold is not None:
            return [x[0] for x in hold]
        else:
            return []

    def get_barcode_details(self, barcode):
        """
        Returns the general barcode details for a barcode
        """
        sql = """SELECT  create_date_time, status, scan_date,
                  sample_postmark_date,
                  biomass_remaining, sequencing_status, obsolete
                  FROM    barcode
                  WHERE barcode = %s"""
        res = self._con.execute_fetchdict(sql, [barcode])
        return res[0] if res else {}

    def get_access_levels(self):
        """Returns tuple of all access levels and ids in the system

        Returns
        -------
        list of tuple of (int, str)
            All access levels in the form (id, name)
        """
        sql = "SELECT access_id, access_name FROM ag.labadmin_access"
        return self._con.execute_fetchall(sql)

    def get_access_levels_user(self, email):
        """Returns tuple of all access levels and ids for a user

        Parameters
        ----------
        email : str
            Email of user to check

        Returns
        -------
        list of tuple of (int, str)
            All access levels in the form (id, name)
        """
        sql = """SELECT access_id, access_name
                 FROM ag.labadmin_access
                 JOIN ag.labadmin_users_access USING (access_id)
                 WHERE email = %s"""
        hold = self._con.execute_fetchall(sql, [email])
        return hold if hold is not None else []

    def alter_access_levels(self, email, levels):
        """Alters existing user's access levels

        Parameters
        ----------
        email : str
            Email of user to alter
        levels : list of int
            List of access level IDs user should now have
        """
        all_levels = set(l[0] for l in self.get_access_levels())
        new_levels = set(levels)
        user_levels = set(l[0] for l in self.get_access_levels_user(email))

        # Delete removed levels
        remove = all_levels - new_levels
        if remove:
            sql = """DELETE FROM ag.labadmin_users_access
                     WHERE email = %s and access_id IN %s"""
            self._con.execute(sql, [email, tuple(remove)])

        # Add new levels
        add = new_levels - user_levels
        if add:
            sql = """INSERT INTO ag.labadmin_users_access (email, access_id)
                     VALUES (%s, %s)"""
            self._con.executemany(sql, [(email, l) for l in add])

    def get_ag_barcode_details(self, barcodes):
        """Retrieve sample, kit, and login details by barcode

        Parameters
        ----------
        barcodes : iterable of str
            The list of barcodes for which to get login details

        Returns
        -------
        dict of dict
            {barcode: {column: value}, ...}
        """
        sql = """SELECT DISTINCT barcode, *
                 FROM ag.ag_kit_barcodes
                 JOIN ag.ag_kit USING (ag_kit_id)
                 LEFT JOIN ag.source_barcodes_surveys USING (barcode)
                 LEFT JOIN ag.ag_login_surveys USING (survey_id)
                 LEFT JOIN ag.ag_login
                 ON (ag.ag_kit.ag_login_id = ag.ag_login.ag_login_id)
                 WHERE barcode in %s"""
        res = self._con.execute_fetchall(sql, [tuple(b[:9] for b in barcodes)])

        parsed = {row[0]: dict(row) for row in res}
        return parsed

    def _survey_sql_to_dataframe(self, sql, barcodes, answer_is_json=False):
        """Query for survey responses, format as a DataFrame

        Parameters
        ----------
        sql : str
            Some query
        barcodes : iterable of str
            The barcodes to query for
        answer_is_json : bool, optional
            If True, the answer field is interpreted to be a json object of the
            form {question: answer}.

        Returns
        -------
        DataFrame
            Columns as: [survey, participant_survey_id, barcode,
            question, answer]. "question" is question shortname.
        """
        result_set = self._con.execute_fetchall(sql, [tuple(barcodes)])

        if answer_is_json:
            obtained = []
            for sid, participant_sid, barcode, q_and_a in result_set:
                for q, a in q_and_a.items():
                    obtained.append((sid, participant_sid, barcode, q, str(a)))
        else:
            obtained = (list(r) for r in result_set)

        return pd.DataFrame(obtained, columns=DATAFRAME_SURVEY_COLUMNS,
                            dtype=str)

    def _get_vioscreen_survey_answers(self, barcodes):
        """Query for vioscreen responses, format as a DataFrame

        Parameters
        ----------
        sql : str
            Some query
        barcodes : iterable of str
            The barcodes to query for

        Returns
        -------
        DataFrame
            Columns as: [survey, participant_survey_id, barcode,
            question, answer]. "question" is question shortname.
        """
        # Get external survey answers and normalize column names
        sql = """SELECT external_survey, survey_id, barcode, answers
                 FROM ag.external_survey_answers
                 JOIN ag.source_barcodes_surveys USING (survey_id)
                 JOIN ag.ag_kit_barcodes USING (barcode)
                 JOIN ag.external_survey_sources
                   USING (external_survey_id)
                 WHERE external_survey = 'Vioscreen' AND barcode IN %s"""

        # the ignored vioscreen entires either do not make sense
        # such as vioscreen_age as the user does not fill this out,
        # or are internal such as vioscreen_username
        vioscreen_to_remove = [u'vioscreen_activity_level', u'vioscreen_age',
                               u'vioscreen_bcodeid', u'vioscreen_bmi',
                               u'vioscreen_dob', u'vioscreen_eer',
                               u'vioscreen_email', u'vioscreen_finished',
                               u'vioscreen_gender', u'vioscreen_height',
                               u'vioscreen_nutrient_recommendation',
                               u'vioscreen_procdate', u'vioscreen_protocol',
                               u'vioscreen_recno', u'vioscreen_scf',
                               u'vioscreen_scfv', u'vioscreen_srvid',
                               u'vioscreen_started', u'vioscreen_time',
                               u'vioscreen_username', u'vioscreen_user_id',
                               u'vioscreen_visit', u'vioscreen_weight']

        df = self._survey_sql_to_dataframe(sql, barcodes, answer_is_json=True)
        to_remove = df.question.isin(vioscreen_to_remove)
        df.drop(to_remove.index[to_remove], inplace=True)
        df.question = [self._convert_header('Vioscreen', q)
                       for q in df.question]
        return df

    def _get_single_survey_answers(self, barcodes):
        """Retrieve SINGLE answers for specific barcodes

        Parameters
        ----------
        barcodes : iterable of str
            The list of barcodes for which metadata will be retrieved

        Returns
        -------
        DataFrame
            Columns as: [survey, participant_survey_id, barcode,
            question, answer]. "question" is question shortname.
        """
        sql = \
            """SELECT S.survey_id AS the_survey,
                      SBS.survey_id AS participant_survey_id,
                      barcode, question_shortname, response
               FROM ag.ag_kit_barcodes
               JOIN ag.source_barcodes_surveys SBS USING (barcode)
               JOIN ag.survey_answers SA USING (survey_id)
               JOIN ag.survey_question USING (survey_question_id)
               JOIN ag.survey_question_response_type USING (survey_question_id)
               JOIN ag.group_questions GQ USING (survey_question_id)
               JOIN ag.surveys S USING (survey_group)
               WHERE survey_response_type='SINGLE'
                   AND (withdrawn IS NULL OR withdrawn != 'Y')
                   AND barcode in %s"""

        return self._survey_sql_to_dataframe(sql, barcodes)

    def _get_multiple_survey_answers(self, barcodes):
        """Retrieve MULTIPLE answers for specific barcodes

        Parameters
        ----------
        barcodes : iterable of str
            The list of barcodes for which metadata will be retrieved

        Returns
        -------
        DataFrame
            Columns as: [survey, participant_survey_id, barcode,
            question, answer]. "question" is question shortname.

        Notes
        -----
        For multiples, the shortname that is used in the returned dict is
        combined with a modified version of the response. Specifically, spaces
        and non-alphanumeric characters are replaced with underscroes, and the
        response is capitalized. E.g., If the person is allergic to "Tree
        nuts", the key in the dict would be "ALLERGIC_TO_TREE_NUTS" (the
        ALLERGIC_TO portion taken from the shortname column of the question
        table).
        """
        # MULTIPLE answers SQL
        sql = \
            """SELECT S.survey_id AS the_survey,
                      SBS.survey_id AS participant_survey_id,
                      barcode, question_shortname,
                      array_agg(response) as responses
               FROM ag.ag_kit_barcodes
               JOIN ag.source_barcodes_surveys SBS USING (barcode)
               JOIN ag.survey_answers USING (survey_id)
               JOIN ag.survey_question USING (survey_question_id)
               JOIN ag.survey_question_response_type USING (survey_question_id)
               JOIN ag.group_questions USING (survey_question_id)
               JOIN ag.surveys S USING (survey_group)
               WHERE survey_response_type='MULTIPLE'
                   AND (withdrawn IS NULL OR withdrawn != 'Y')
                   AND barcode in %s
               GROUP BY S.survey_id, barcode, SBS.survey_id, question_shortname"""

        query = self._survey_sql_to_dataframe(sql, barcodes)

        # Get all the possible responses for the multiple selection questions
        sql = \
            """SELECT question_shortname, response
               FROM survey_question
               JOIN survey_question_response_type USING (survey_question_id)
               JOIN survey_question_response USING (survey_question_id)
               WHERE survey_response_type = 'MULTIPLE'"""

        headers = {}
        for question, response in self._con.execute_fetchall(sql):
            if question not in headers:
                headers[question] = {}

            # Format the question / response to a Qiita compatible shortname
            # For example, FERMENTED_CONSUMED and Other, Cider, etc to
            # FERMENTED_CONSUMED_OTHER, FERMENTED_CONSUMED_CIDER
            response = response.replace(" ", "_")
            response = sub('\W', '', response)
            header = '_'.join([question, response]).upper()
            headers[question][response] = header

        # Each row needs to expand to N rows, where N is the number
        # of multiple responses. Or rather, we need to translate question
        # which represents multiple checkboxes, such as the selection choises
        # for FERMENTED_CONSUMED, into individual responses. The strategy
        # we're going to use is to pull out subsets of the DataFrame per
        # "multiple" question, mark those rows for deletion, create
        # N rows per individual response, and finally delete the rows and
        # concatenate our expanded responses together.
        to_drop = pd.Int64Index([])
        new_rows = []

        for question in query['question'].unique():
            question_grp = query[query['question'] == question].copy()
            to_drop = to_drop.append(question_grp.index)

            response_to_header = headers[question]
            for _, row in question_grp.iterrows():
                row_answer = row['answer']

                for response, header in response_to_header.items():
                    # row answer is a tuple
                    new_answer = response in row_answer
                    new_answer = 'true' if new_answer else 'false'

                    new_row = row.copy()
                    new_row['question'] = header
                    new_row['answer'] = new_answer
                    new_rows.append(new_row)

        if new_rows or to_drop.size > 0:
            query.drop(to_drop, inplace=True)
            new_rows = pd.DataFrame(new_rows)
            query = pd.concat([query, new_rows], ignore_index=True,
                              verify_integrity=True)

        return query

    def _get_other_survey_answers(self, barcodes):
        """Retrieve STRING / TEXT answers for specific barcodes

        Parameters
        ----------
        barcodes : iterable of str
            The list of barcodes for which metadata will be retrieved

        Returns
        -------
        DataFrame
            Columns as: [survey, participant_survey_id, barcode,
            question, answer]. "question" is question shortname.
        """
        sql = \
            """SELECT S.survey_id AS the_survey,
                      SBS.survey_id AS participant_survey_id,
                      barcode, question_shortname, response
               FROM ag.ag_kit_barcodes
               JOIN ag.source_barcodes_surveys SBS USING (barcode)
               JOIN ag.survey_answers_other SA USING (survey_id)
               JOIN ag.survey_question USING (survey_question_id)
               JOIN ag.survey_question_response_type USING (survey_question_id)
               JOIN ag.group_questions GQ USING (survey_question_id)
               JOIN ag.surveys S USING (survey_group)
               WHERE survey_response_type IN ('STRING', 'TEXT')
                   AND (withdrawn IS NULL OR withdrawn != 'Y')
                   AND barcode IN %s"""

        query = self._survey_sql_to_dataframe(sql, barcodes)

        # clean up garbage from json responses
        if len(query):
            query['answer'] = [unicode(a, 'utf-8').strip('"\'[]_,\t\r\n\\/ ')
                               for a in query['answer']]

        return query

    def _clean_and_ambiguous_barcodes(self, barcodes):
        """Strip suffixes from barcodes"""
        # find special case barcodes with appended info and store them
        # e.g., 000001234A
        ambig = defaultdict(list)
        barcodes_clean = set()
        for b in barcodes:
            clean = b[:9]
            barcodes_clean.add(clean)

            if len(b) > 9:
                ambig[clean].append(b)
            if len(b) < 9:
                raise ValueError("%s is shorter than 9 characters." % b)

        return (tuple(sorted(barcodes_clean)), ambig)

    def get_surveys(self, barcodes):
        """Retrieve surveys for specific barcodes

        Parameters
        ----------
        barcodes : iterable of str
            The list of barcodes for which metadata will be retrieved

        Returns
        -------
        DataFrame
            Columns as: [survey, participant_survey_id, barcode,
            question, answer]. "question" is question shortname.
        """
        clean, ambig = self._clean_and_ambiguous_barcodes(barcodes)

        single = self._get_single_survey_answers(clean)
        multiple = self._get_multiple_survey_answers(clean)
        other = self._get_other_survey_answers(clean)
        vioscreen = self._get_vioscreen_survey_answers(clean)

        result = pd.concat([single, other, multiple, vioscreen],
                           ignore_index=True,
                           verify_integrity=True)

        # resolve ambiguities (e.g., 000000000 -> 000000000A and 000000000B)
        to_drop = pd.Int64Index([])
        new_rows = []
        for singular, replicate in ambig.items():
            # a requested barcode does not responses
            if singular not in result.barcode.unique():
                continue

            rows = result[result['barcode'] == singular].copy()
            to_drop = to_drop.append(rows.index)
            rows.reset_index(drop=True, inplace=True)

            for rep in replicate:
                new_rep_rows = rows.copy()
                new_rep_rows.loc[:, 'barcode'] = rep
                new_rows.append(new_rep_rows)

        if new_rows or to_drop.size > 0:
            result.drop(to_drop, inplace=True)
            result = pd.concat([result] + new_rows, ignore_index=True,
                               verify_integrity=True)

        return result

    def _months_between_dates(self, d1, d2):
        """Calculate the number of months between two dates

        Parameters
        ----------
        d1 : datetime
            First date
        d2 : datetime
            Second date

        Raises
        ------
        ValueError
            if the first date is greater than the second date

        Notes
        -----
        - Assumes the first date d1 is not greater than the second date d2
        - Ignores the day (uses only year and month)
        """
        if d1 > d2:
            raise ValueError("First date must not be greater than the second")

        # Calculate the number of 12-month periods between the years
        return (d2.year - d1.year) * 12 + (d2.month - d1.month)

    def _geocode(self, barcode, zipcode, country, zip_lookup, country_lookup):
        """Adds geocoding information to the barcoe for pulldown"""
        # for a proper lookup in the dict, zipcode must be encoded as utf-8
        key_zipcode = zipcode.encode('utf-8')
        print(zip_lookup[key_zipcode])
        #try:
        barcode['LATITUDE'] = zip_lookup[key_zipcode][country][0]
        barcode['LONGITUDE'] = zip_lookup[key_zipcode][country][1]
        barcode['ELEVATION'] = zip_lookup[key_zipcode][country][2]
        barcode['STATE'] = zip_lookup[key_zipcode][country][3]
        barcode['COUNTRY'] = country_lookup[country]
        barcode['GEO_LOC_NAME'] = ':'.join(
            [barcode['COUNTRY'], barcode['STATE']])
        #except KeyError:
        #    # geocode unknown zip/country combo and add to
        #    # zipcode table & lookup dict
        #    info = self.get_geocode_zipcode(zipcode, country)
        #    if info.lat is not None:
        #        barcode['LATITUDE'] = "%.1f" % info.lat
        #        barcode['LONGITUDE'] = "%.1f" % info.long
        #        barcode['ELEVATION'] = "%.1f" % info.elev
        #        barcode['STATE'] = info.state
        #        barcode['COUNTRY'] = country_lookup[info.country]
        #        barcode['GEO_LOC_NAME'] = ':'.join(
        #            [barcode['COUNTRY'], barcode['STATE']])
        #        # Store in dict so we don't geocode again
        #        zip_lookup[key_zipcode][country] = (
        #            round(info.lat, 1), round(info.long, 1),
        #            round(info.elev, 1), info.state)
        #    else:
        #        barcode['LATITUDE'] = 'Unspecified'
        #        barcode['LONGITUDE'] = 'Unspecified'
        #        barcode['ELEVATION'] = 'Unspecified'
        #        barcode['STATE'] = 'Unspecified'
        #        barcode['COUNTRY'] = 'Unspecified'
        #        barcode['GEO_LOC_NAME'] = 'Unspecified'
        #        # Store in dict so we don't geocode again
        #        zip_lookup[key_zipcode][country] = (
        #            'Unspecified', 'Unspecified', 'Unspecified',
        #            'Unspecified')
        return barcode

    def _construct_country_lookup(self):
        country_sql = "SELECT country, EBI from ag.iso_country_lookup"
        country_lookup = dict(self._con.execute_fetchall(country_sql))
        # Add for scrubbed testing database
        country_lookup['REMOVED'] = 'REMOVED'
        return country_lookup

    def _construct_zip_lookup(self, full=False):
        """Construct a map associating a zipcode to location details

        Parameters
        ----------
        full : bool
            If true, do not truncate latitude and longitude

        Returns
        -------
        dict of dict
            {zipcode: {country: [latitude, longitude, elevation, state]}}
        """
        # tuples are latitude, longitude, elevation, state
        if full:
            zipcode_sql = """SELECT UPPER(zipcode), country,
                                 latitude::numeric,
                                 longitude::numeric,
                                 elevation::numeric, state
                             FROM zipcodes"""
        else:
            zipcode_sql = """SELECT UPPER(zipcode), country,
                                 round(latitude::numeric, 1),
                                 round(longitude::numeric,1),
                                 round(elevation::numeric, 1), state
                             FROM zipcodes"""

        zip_lookup = defaultdict(dict)

        def _decode_zip_lookup(item):
            if item is None:
                return "Not provided"
            elif isinstance(item, (str, unicode)):
                return item.decode('utf-8')
            else:
                return item

        for row in self._con.execute_fetchall(zipcode_sql):
            zip_lookup[row[0]][row[1]] = map(_decode_zip_lookup, row[2:])

        return zip_lookup

    def _smooth_survey_pets(self, md):
        """Add static entries for any unique pet survey"""
        sid = 2
        pets = md[md.survey == sid]
        new_rows = self._smooth_survey_host_invariant(pets, is_human=False)
        return pd.concat([md, new_rows])

    def _smooth_survey_host_invariant(self, md, is_human):
        if is_human:
            host_taxid = '9606'
            scientific_name = 'Homo sapiens'
            env_feature = 'human-associated habitat'
            host_common_name = 'Not provided'
            descrip = "Not provided"  # NOTE: Down stream; based on sample type
        else:
            host_taxid = 'Not provided'
            scientific_name = 'Not provided'
            env_feature = 'animal-associated habitat'
            host_common_name = 'human'
            descrip = "American Gut Project Animal sample"

        added = set()
        new_rows = []
        for idx, row in md.iterrows():
            sid = row['survey']
            barcode = row['barcode']
            part_survey_id = row['participant_survey_id']

            if (barcode, part_survey_id) in added:
                continue
            else:
                added.add((barcode, part_survey_id))

            filler = (sid, part_survey_id, barcode)
            new_rows.append(filler + ('ANONYMIZED_NAME', barcode))
            new_rows.append(filler + ('HOST_TAXID', host_taxid))
            new_rows.append(filler + ('SCIENTIFIC_NAME', scientific_name))
            new_rows.append(filler + ('HOST_SUBJECT_ID', barcode))
            new_rows.append(filler + ('TITLE', 'American Gut Project'))
            new_rows.append(filler + ('ALTITUDE', 'Not applicable'))
            new_rows.append(filler + ('ASSIGNED_FROM_GEO', 'true'))
            new_rows.append(filler + ('ENV_BIOME', 'dense settlement biome'))
            new_rows.append(filler + ('ENV_FEATURE', env_feature))
            new_rows.append(filler + ('DEPTH', 'Not applicable'))
            new_rows.append(filler + ('DESCRIPTION', descrip))
            new_rows.append(filler + ('DNA_EXTRACTED', 'true'))
            new_rows.append(filler + ('HOST_COMMON_NAME', host_common_name))
            new_rows.append(filler + ('PHYSICAL_SPECIMEN_REMAINING', 'true'))
            new_rows.append(filler + ('PHYSICAL_SPECIMEN_LOCATION', 'UCSDMI'))

        return pd.DataFrame(new_rows, columns=DATAFRAME_SURVEY_COLUMNS,
                            dtype=str)

    def _smooth_survey_yesno(self, md):
        """Inplace cast yes/no responses to true / false"""
        if len(md):
            remap = {'yes': 'true',
                    'true': 'true',
                    'false': 'false',
                    'no': 'false'}
            revised = [remap.get(a.lower(), a)
                       for a in md['answer']]
            md['answer'] = revised
        return md

    def _human_normalize_numeric(self, df):
        """Convert fields that strings to numeric"""
        for field in ('HEIGHT_CM', 'WEIGHT_KG'):
            df[field] = pd.to_numeric(df[field], errors='coerce')
            df.loc[df[field].isnull(), field] = 'Not provided'
        return df

    def _human_normalize_height(self, df):
        """Normalize height to CM"""
        inches = df['HEIGHT_UNITS'] == 'inches'
        height = ~(df['HEIGHT_CM'].isin(['', np.nan, 'Not provided']))
        to_normalize = inches & height

        subset = df[to_normalize]
        df.loc[to_normalize, 'HEIGHT_CM'] = subset['HEIGHT_CM'] * 2.54
        df.loc[inches, 'HEIGHT_UNITS'] = 'centimeters'
        return df

    def _human_normalize_weight(self, df):
        """Normalize weight to KG"""
        pounds = df['WEIGHT_UNITS'] == 'pounds'
        weight = ~(df['WEIGHT_KG'].isin(['', np.nan, 'Not provided']))
        to_normalize = pounds & weight

        subset = df[to_normalize]
        df.loc[to_normalize, 'WEIGHT_KG'] = subset['WEIGHT_KG'] / 2.20462
        df.loc[pounds, 'WEIGHT_UNITS'] = 'kilograms'

        return df

    def _human_create_bmi(self, df):
        """Create BMI, BMI_CORRECTED and BMI_CAT"""
        def bmi(row):
            weight = row['WEIGHT_KG']
            height = row['HEIGHT_CM']

            if height == 'Not provided' or weight == 'Not provided':
                return 'Not provided'
            elif height == 0 or weight == 0:
                return 'Not provided'
            else:
                return float(weight) / (float(height) / 100)**2

        def corrected(row):
            bmi = row['BMI']
            return correct_bmi(bmi, 'Not provided')

        def category(row):
            bmi = row['BMI']
            return categorize_bmi(bmi, 'Not provided')

        df['BMI'] = df.apply(bmi, axis=1)
        df['BMI_CORRECTED'] = df.apply(corrected, axis=1)
        df['BMI_CAT'] = df.apply(category, axis=1)

        return df

    def _human_create_age_years(self, df):
        """Create AGE_YEARS"""
        def age_years(row):
            month = row['BIRTH_MONTH']
            year = row['BIRTH_YEAR']

            if month == 'Not provided' or year == 'Not provided':
                return 'Not provided'
            else:
                col_year = row['COLLECTION_YEAR']
                col_month = row['COLLECTION_MONTH']
                col_day = row['COLLECTION_DAY']

                birthdate = datetime(int(year), int(month), 1)
                coldate = datetime(col_year, col_month, col_day)

                age_in_months = self._months_between_dates(birthdate, coldate)

                return str(int(age_in_months / 12.0))

        df['AGE_YEARS'] = df.apply(age_years, axis=1)
        return df

    def _human_create_age_corrected(self, df):
        """Create AGE_CORRECTED and AGE_CAT"""
        def age_corrected(row):
            years = row['AGE_YEARS']
            height = row['HEIGHT_CM']
            weight = row['WEIGHT_KG']
            alc = row['ALCOHOL_CONSUMPTION']
            return correct_age(years, height, weight, alc, 'Not provided')

        def age_category(row):
            corrected = row['AGE_CORRECTED']
            return categorize_age(corrected, 'Not provided')

        df['AGE_CORRECTED'] = df.apply(age_corrected, axis=1)
        df['AGE_CAT'] = df.apply(age_category, axis=1)

        return df

    def _human_create_sex(self, df):
        """Create the SEX category"""
        def sex(row):
            g = row['GENDER']
            if g is not None:
                return g.lower()
            else:
                return 'Not provided'

        df['SEX'] = df.apply(sex, axis=1)
        return df

    def _human_create_ibd_diagnosis(self, df):
        """Create the IBD_DIAGNOSIS column"""
        mapping = {"Ileal Crohn's Disease": "Crohn's disease",
                   "Colonic Crohn's Disease" : "Crohn's disease",
                   "Ileal and Colonic Crohn's Disease": "Crohn's disease",
                   'Ulcerative colitis': 'Ulcerative colitis'}
        def ibd(row):
            obs = row['IBD_DIAGNOSIS_REFINED']
            return mapping.get(obs, 'Not provided')

        df['IBD_DIAGNOSIS'] = df.apply(ibd, axis=1)
        return df

    def _human_create_alcohol_consumption(self, df):
        """Create inferred ALCOHOL_CONSUMPTION field"""
        def alcohol(row):
            return categorize_etoh(row['ALCOHOL_FREQUENCY'], 'Not provided')

        df['ALCOHOL_CONSUMPTION'] = df.apply(alcohol, axis=1)
        return df

    def _human_create_collection_season(self, df):
        def season(row):
            return season_lookup.get(row['COLLECTION_MONTH'], 'Not provided')

        df['COLLECTION_SEASON'] = df.apply(season, axis=1)
        return df

    def _human_create_economic_census_regions(self, df):
        """Create the inferred CENSUS_REGION and ECONOMIC_REGION categories"""
        missing = {'Census_1': 'Not provided', 'Economic': 'Not provided'}

        def census(row):
            state = row.get('STATE', 'Not provided')
            return regions_by_state.get(state, missing)['Census_1']

        def economic(row):
            state = row.get('STATE', 'Not provided')
            return regions_by_state.get(state, missing)['Economic']

        df['CENSUS_REGION'] = df.apply(census, axis=1)
        df['ECONOMIC_REGION'] = df.apply(economic, axis=1)

        return df

    def _human_create_subset_age(self, df):
        """Create the inferred SUBSET_AGE column"""
        if 'AGE_YEARS' not in df.columns:
            return df

        def test(row):
            val = row['AGE_YEARS']
            return not np.isnan(val) and (19 < val < 70)

        df['SUBSET_AGE'] = df.apply(test, axis=1)
        return df

    def _human_create_subset_bmi(self, df):
        """Create the inferred SUBSET_BMI column"""
        criteria = {'BMI_CORRECTED': (18.5, 30)}
        if 'BMI_CORRECTED' not in df.columns:
            return df

        def test(row):
            val = row['BMI_CORRECTED']
            return not np.isnan(val) and (18.5 < val < 30)

        df['SUBSET_BMI'] = df.apply(test, axis=1)
        return df

    def _human_create_subset_diabetes(self, df):
        """Create the inferred SUBSET_DIABETES column"""
        criteria = {'DIABETES': 'I do not have this condition'}
        return self._human_create_subset_column(df, 'SUBSET_DIABETES',
                                                criteria)

    def _human_create_subset_ibd(self, df):
        """Create the inferred SUBSET_IBD column"""
        criteria = {'IBD': 'I do not have this condition'}
        return self._human_create_subset_column(df, 'SUBSET_IBD', criteria)

    def _human_create_subset_antibiotic_history(self, df):
        """Create the inferred SUBSET_ANTIBIOTIC_HISTORY column"""
        criteria = {'ANTIBIOTIC_HISTORY':
                        'I have not taken antibiotics in the past year.'}
        return self._human_create_subset_column(df,
                                                'SUBSET_ANTIBIOTIC_HISTORY',
                                                criteria)

    def _human_create_subset_healthy(self, df):
        """Create the inferred SUBSET_HEALTHY column"""
        criteria = {'SUBSET_AGE': True,
                    'SUBSET_DIABETES': True,
                    'SUBSET_IBD': True,
                    'SUBSET_ANTIBIOTIC_HISTORY': True,
                    'SUBSET_BMI': True}
        return self._human_create_subset_column(df, 'SUBSET_HEALTHY', criteria)

    def _human_create_subset_column(self, df, result, criteria):
        """Return a new series based on exact criteria

        Parameters
        ----------
        df : pd.DataFrame
            The frame to operate on
        result : str
            The new column name to store into.
        criteria : dict
            The key corresponds to a column in the df, the value to the
            condition which must be met.

        Returns
        -------
        pd.DataFrame
        A new series of bool, where True indicates the criteria were met
        or False if the criteria were not met is added to the DataFrame.
        If any of the criteria columns are not found, then the DataFrame
        is returned unmodified.
        """
        if not set(criteria).issubset(set(df.columns)):
            return df

        def test(row):
            observed = [not np.isnan(row[col]) and row[col] == exp
                        for col, exp in criteria.items()]
            return np.all(observed)

        df[result] = df.apply(test, axis=1)
        return df

    def _smooth_survey_human(self, md):
        """Add constant details, and normalize where necessary"""
        sid = 1
        humans = md[md.survey == sid]
        if len(humans) == 0:
            return md

        with_invariant = self._smooth_survey_host_invariant(humans, is_human=True)
        humans = pd.concat([humans, with_invariant])

        # use of the str.cat from @JohnE in https://stackoverflow.com/a/26476883/19741
        pivot = pd.pivot_table(humans, values='answer',
                               index=['survey', 'participant_survey_id',
                                      'barcode'],
                               columns=['question'], aggfunc=lambda x: x.str.cat())

        creations = [self._human_create_bmi,
                     self._human_create_ibd_diagnosis,
                     self._human_create_economic_census_regions,
                     self._human_create_alcohol_consumption,
                     self._human_create_collection_season,
                     self._human_create_sex,
                     self._human_create_age_years,
                     self._human_create_age_corrected,

                     self._human_create_subset_antibiotic_history,
                     self._human_create_subset_ibd,
                     self._human_create_subset_diabetes,
                     self._human_create_subset_age,
                     self._human_create_subset_bmi,
                     self._human_create_subset_healthy]

        for f in creations:
            pivot = f(pivot)

        melted = pd.melt(pivot.reset_index(), value_name='answer',
                         id_vars=['survey', 'participant_survey_id',
                                  'barcode'])
        return pd.concat([md, melted])

        ### set description by sample type
        # Human survey (id 1)

        ###HUMAN_CATEGORIES_TO_FUNCTION = {
        ###    ('HEIGHT_CM', ): _human_normalize_numeric,
        ###    ('WEIGHT_KG', ): _human_normalize_numeric,
        ###    ('GENDER', ): _human_create_sex,
        ###    ('HEIGHT_UNITS', 'HEIGHT_CM'): _human_normalize_height,
        ###    ('WEIGHT_UNITS', 'WEIGHT_KG'): _human_normalize_weight,
        ###    ('HEIGHT_CM', 'WEIGHT_KG'): _human_create_bmi,
        ###    ('BIRTH_MONTH', 'BIRTH_YEAR', 'sample_date'): _human_create_age_years_and_subset,
        ###    ('ZIP_CODE', 'country'): _human_create_geocode,
        ###    ('sample_time', 'sample_date'): _human_create_collection_timestamp,
        ###    ('site_sampled', ): _human_set_site_specific_metadata,
        ###    ('IBD_DIAGNOSIS_REFINED', ): _human_create_ibd_diagnosis,
        ###    ('ALCOHOL_FREQUENCY', ): _human_create_alcohol_consumption,
        ###    ('sample_date', ): _human_create_collection_season,
        ###    ('STATE', ): _human_create_economic_census_regions,
        ###    ('DIABETES', ): _human_create_subset_diabetes,
        ###    ('IBD', ): _human_create_subset_ibd,
        ###    ('ANTIBIOTIC_HISTORY', ): _human_create_subset_antibiotic_history,
        ###    ('SUBSET_AGE', 'SUBSET_DIABETES', 'SUBSET_IBD', 'SUBSET_ANTIBIOTIC_HISTORY', 'SUBSET_BMI'): _human_create_subset_healthy,
        ###    ('AGE_YEARS', 'HEIGHT_CM', 'WEIGHT_KG', 'ALCOHOL_CONSUMPTION'): _human_create_age_corrected_and_category,

        ###for barcode, responses in md[1].items():
        ###    bc_info = barcode_info[barcode[:9]]
        ###    try:
        ###        participant_name = dupes_lookup.get(
        ###            md[1][barcode]['SURVEY_ID'],
        ###            bc_info['participant_name']).lower()

        ###        md[1][barcode]['HOST_SUBJECT_ID'] = sha512(
        ###            bc_info['ag_login_id'] + participant_name).hexdigest()
        ###        md[1][barcode]['PUBLIC'] = 'Yes'


        ###        # Get rid of columns not wanted for pulldown
        ###        if not full:
        ###            for col in ebi_remove:
        ###                try:
        ###                    del md[1][barcode][col]
        ###                except KeyError:
        ###                    # Column doesn't exist already for survey
        ###                    # (retired), so no removal needed
        ###                    pass

        ###        # Add the external surveys
        ###        if unknown_external:
        ###            md[1][barcode].update(external.get(md[1][barcode][
        ###                'SURVEY_ID'], unknown_external))

        ###        self._sync_with_data_dictionary(md[1][barcode], not_provided,
        ###                                        not_applicable)
        ###    except Exception as e:
        ###        # Add barcode to error and remove from metadata info
        ###        if isinstance(e, SSLError):
        ###            errors[barcode] = repr(e)
        ###        else:
        ###            errors[barcode] = str(e.message.encode('utf-8'))
        ###        del md[1][barcode]


    def _integrate_barcode_information(self, md, barcode_info, full=False):
        """Integrate barcoe information (e.g., sample time, type, etc)"""
        zip_lookup = self._construct_zip_lookup(full)
        country_lookup = self._construct_country_lookup()

        distinct = md[['survey',
                       'participant_survey_id',
                       'barcode']]
        distinct = distinct[~distinct.duplicated()]

        to_add = []
        for _, row in distinct.iterrows():
            filler = list(row)

            bc_info = barcode_info[row['barcode']]
            sample_invariant = md_lookup[bc_info['site_sampled']]

            for item in sample_invariant.items():
                to_add.append(filler + list(item))

            dt = bc_info['sample_date']
            to_add.append(filler + ['COLLECTION_YEAR', dt.year])
            to_add.append(filler + ['COLLECTION_MONTH', dt.month])
            to_add.append(filler + ['COLLECTION_DAY', dt.day])
            to_add.append(filler + ['COLLECTION_DATTE',
                                    dt.strftime('%m/%d/%Y')])

            tt = bc_info['sample_time']
            if not tt:
                tt = datetime.time(0, 0)
            to_add.append(filler + ['COLLECTION_HOUR', tt.hour])
            to_add.append(filler + ['COLLECTION_MINUTE', tt.minute])
            to_add.append(filler + ['COLLECTION_TIME',
                                    tt.strftime('%H:%M')])

            ts = datetime.combine(dt, tt)
            to_add.append(filler + ['COLLECTION_TIMESTAMP',
                                    ts.strftime('%m/%d/%Y %H:%M')])

            name = bc_info['participant_name']
            loginid = bc_info['ag_login_id']
            hsi = sha512(loginid + name).hexdigest()
            to_add.append(filler + ['HOST_SUBJECT_ID', hsi])

            # so not excited by this. this monkeying is necessary to reuse
            # existing geocoder instead of rewriting it.

            # the geocoder does not actually work. the country information
            # contained in zip_lookup is inconsistent with the information
            # in country look up. it seems that the entirety of the logic
            # in _geocoder that was used was within its "except" block.
            # the _geocoder needs to be rewritten. presently marking this
            # as out of scope.
            #zipcode = bc_info['ZIP_CODE'].upper()
            #country = bc_info['country']
            #inout = {'STATE': bc_info['STATE']}
            #self._geocode(inout, zipcode, country, zip_lookup, country_lookup)
            #for item in inout.items():
            #    to_add.append(filler + list(item))

        to_add = pd.DataFrame(to_add, columns=['survey',
                                               'participant_survey_id',
                                               'barcode',
                                               'question',
                                               'answer'])
        return pd.concat([md, to_add], ignore_index=True)

    def format_survey_data(self, md, full=False):  # noqa
        """Modifies barcode metadata to include all columns and correct units

        Specifically, this function:
        - corrects height and weight to be in the same units (cm and kg,
          respectively)
        - Adds AGE_MONTHS and AGE_YEARS columns
        - Adds standard columns for EBI and MiMARKS

        Parameters
        ----------
        md : DataFrame
            Columns as: [survey, participant_survey_id, barcode,
            question, answer]. "question" is question shortname.
        full : bool
            Whether to pull full or filtered answers. Default filtered (False)

        Returns
        -------
        dict of dict of dict
            the formatted metadata
        list of tuple of str
            The barcode and error message if something failed
        """
        not_provided = 'Not provided'
        not_applicable = 'Not applicable'
        errors = {}

        # get barcode information
        all_barcodes = set(md['barcode'].unique())
        barcode_info = self.get_ag_barcode_details(all_barcodes)

        #dupes_sql = """SELECT duplicate_survey_id, participant_name
        #               FROM ag.duplicate_consents dc
        #               JOIN ag.ag_login_surveys als USING (ag_login_id)
        #               WHERE  dc.main_survey_id = als.survey_id"""
        #dupes_lookup = dict(self._con.execute_fetchall(dupes_sql))

        md = self._integrate_barcode_information(md, barcode_info, full)
        md = self._smooth_survey_yesno(md)
        md = self._smooth_survey_pets(md)
        md = self._smooth_survey_human(md)

        ### need general geocoding for both pets / human
                #md[2][barcode] = self._geocode(md[2][barcode], zipcode, country,
                 #                              zip_lookup, country_lookup)

        for survey in md:
            for barcode, responses in md[survey].items():
                for c, v in md[survey][barcode].items():
                    if isinstance(v, (str, unicode)) and v.lower() == 'unspecified':
                        md[survey][barcode][c] = not_provided

        return md, errors

    def _sync_with_data_dictionary(self, barcode_metadata, not_provided,
                                   not_applicable):
        """Remap responses where necessary to conform to submitted data dict

        Parameters
        ----------
        barcode_metadata : dict
            A dict of the metadata responses associated with the barcode.
        not_provided : str
            A string to use for values that are not provided.
        not_applicable : str
            A string to use for values that are not applicable.

        Notes
        -----
        This method operates in place
        """
        boolean_remap = {
            'pregnant', 'public', 'roommates_in_study',
            'seasonal_allergies', 'softener',
            'breastmilk_formula_ensure',
            'specialized_diet_exclude_dairy',
            'specialized_diet_exclude_nightshades',
            'specialized_diet_exclude_refined_sugars',
            'specialized_diet_fodmap', 'specialized_diet_halaal',
            'specialized_diet_i_do_not_eat_a_specialized_diet',
            'specialized_diet_kosher',
            'specialized_diet_modified_paleo_diet',
            'specialized_diet_other_restrictions_not_described_here',
            'specialized_diet_paleodiet_or_primal_diet',
            'specialized_diet_raw_food_diet',
            'specialized_diet_unspecified',
            'specialized_diet_westenprice_or_other_lowgrain_low_processed_food_diet',  # noqa
            'subset_age', 'subset_antibiotic_history', 'subset_bmi',
            'subset_diabetes', 'subset_healthy', 'subset_ibd',
            'tonsils_removed', 'multivitamin', 'nail_biter',
            'non_food_allergies_beestings',
            'non_food_allergies_drug_eg_penicillin',
            'non_food_allergies_pet_dander',
            'non_food_allergies_poison_ivyoak', 'non_food_allergies_sun',
            'non_food_allergies_unspecified', 'other_supplement_frequency',
            'pets_other', 'acne_medication', 'acne_medication_otc',
            'alcohol_consumption', 'alcohol_types_beercider',
            'alcohol_types_red_wine', 'alcohol_types_sour_beers',
            'alcohol_types_spiritshard_alcohol',
            'alcohol_types_unspecified', 'alcohol_types_white_wine',
            'allergic_to_i_have_no_food_allergies_that_i_know_of',
            'allergic_to_other', 'allergic_to_peanuts',
            'allergic_to_shellfish', 'allergic_to_tree_nuts',
            'allergic_to_unspecified', 'appendix_removed',
            'assigned_from_geo', 'cat',
            'chickenpox', 'csection', 'dna_extracted', 'dog',
            'has_physical_specimen', 'lactose', 'livingwith',
            'lowgrain_diet_type', 'mental_illness',
            'consume_animal_products_abx',
            'mental_illness_type_anorexia_nervosa',
            'mental_illness_type_bipolar_disorder',
            'mental_illness_type_bulimia_nervosa',
            'mental_illness_type_depression',
            'mental_illness_type_ptsd_posttraumatic_stress_disorder',
            'mental_illness_type_schizophrenia',
            'mental_illness_type_substance_abuse',
            'mental_illness_type_unspecified'}

        boolean_remap_vals = {'yes': u'true',
                              'no': u'false',
                              'false': u'false',
                              'true': u'true',
                              'not sure': u'Not sure'}

        # values in this category need to be transformed except this one
        boolean_remap_vals['i eat both solid food and formula/breast milk'] = \
            'I eat both solid food and formula/breast milk'

        null_remap = {'not applicable': not_applicable,
                      'not provided': not_provided,
                      'unspecified': not_provided}

        for k, v in barcode_metadata.items():
            v = str(v).lower()
            if v in null_remap:
                new_v = null_remap[v]
                barcode_metadata[k] = new_v

            elif k.lower() in boolean_remap:
                new_v = boolean_remap_vals.get(v)
                if new_v is None:
                    raise ValueError("Unexpected value, %s -> %s" % (k, v))
                barcode_metadata[k] = new_v

    def format_environmental(self, barcodes):
        """Format the environemntal data pulldown metadata

        Parameters
        ----------
        barcodes : list of (barcode, env sampled)
            List of tuples of barcode and the environment sampled

        Returns
        -------
        str
            Formatted tsv metadata for the environmental samples
        """
        md = {}
        errors = {}
        barcode_info = self.get_ag_barcode_details(
            [b[0][:9] for b in barcodes])
        # tuples are latitude, longitude, elevation, state
        zipcode_sql = """SELECT UPPER(zipcode), country,
                             round(latitude::numeric, 1),
                             round(longitude::numeric,1),
                             round(elevation::numeric, 1), state
                         FROM zipcodes"""
        zip_lookup = defaultdict(dict)
        for row in self._con.execute_fetchall(zipcode_sql):
            zip_lookup[row[0]][row[1]] = map(
                lambda x: x if x is not None else 'Unspecified', row[2:])

        country_sql = "SELECT country, EBI from ag.iso_country_lookup"
        country_lookup = dict(self._con.execute_fetchall(country_sql))
        # Add for scrubbed testing database
        country_lookup['REMOVED'] = 'REMOVED'

        for barcode, env in barcodes:
            # Not using defaultdict so we don't ever allow accidental insertion
            # of unknown barcodes
            md[barcode] = {}
            # Add info from constants dict
            try:
                md[barcode].update(env_lookup[env])
                # Invariant information
                md[barcode]['TITLE'] = 'American Gut Project'
                md[barcode]['ASSIGNED_FROM_GEO'] = 'true'
                md[barcode]['PHYSICAL_SPECIMEN_REMAINING'] = 'true'
                md[barcode]['PHYSICAL_SPECIMEN_LOCATION'] = 'UCSDMI'

                # Barcode specific information
                specific_info = barcode_info[barcode[:9]]

                md[barcode]['ANONYMIZED_NAME'] = barcode
                md[barcode]['HOST_SUBJECT_ID'] = barcode

                # Geolocate based on kit information, since no other
                # geographic info available
                zipcode = specific_info['zip'].upper()
                country = specific_info['country']
                md[barcode] = self._geocode(md[barcode], zipcode, country,
                                            zip_lookup, country_lookup)

                md[barcode]['COLLECTION_DATE'] = \
                    specific_info['sample_date'].strftime('%m/%d/%Y')
                if specific_info['sample_time']:
                    md[barcode]['COLLECTION_TIME'] = \
                        specific_info['sample_time'].strftime('%H:%M')
                else:
                    # If no time data, show unspecified and default to midnight
                    md[barcode]['COLLECTION_TIME'] = 'Unspecified'
                    specific_info['sample_time'] = time(0, 0)

                md[barcode]['COLLECTION_TIMESTAMP'] = datetime.combine(
                    specific_info['sample_date'],
                    specific_info['sample_time']).strftime('%m/%d/%Y %H:%M')
            except Exception as e:
                del md[barcode]
                errors[barcode] = str(e)
                continue
        return md, errors

    def participant_names(self):
        """Retrieve the participant names for the given barcodes

        Returns
        -------
        list of tuple
            (barcode, participant name)
        """
        sql = """SELECT barcode, participant_name
                 FROM ag.ag_kit_barcodes
                 JOIN ag.source_barcodes_surveys USING (barcode)
                 JOIN ag.ag_login_surveys USING (survey_id)
                 WHERE participant_name IS NOT NULL"""
        return self._con.execute_fetchall(sql)

    def _convert_header(self, survey, header):
        return converter.camel_to_snake('_'.join(
            [survey.replace(' ', '_'), header])).upper()

    def pulldown(self, barcodes, blanks=None,  # noqa
                 full=False):
        """Pulls down AG metadata for given barcodes

        Parameters
        ----------
        barcodes : list of str
            Barcodes to pull metadata down for
        blanks : list of str, optional
            Names for the blanks to add. Default None
            Blanks added to survey 1
        external : list of str, optional
            External surveys to add to the pulldown, default None
        full : bool, optional
            If True do a full pulldown, otherwise do an EBI-cleaned pulldown.
            Default False.

        Returns
        -------
        metadata : dict of str
            Tab delimited qiita sample template, keyed to survey ID it came
            from
        failures : dict
            Barcodes unable to pull metadata down, in the form
            {barcode: reason, ...}
        """
        all_results = {}
        errors = {}
        all_survey_info = self.get_surveys(barcodes)
        if len(all_survey_info) > 0:
            all_results, errors = self.format_survey_data(all_survey_info,
                                                          full)
        #### ###
        # pick up from here
        ####
        ####
        ####
        ####
        ####
        ####
        ####
        ####
        ####
        ####
        ####
        # Do the pulldown for the environmental samples
        sql = """SELECT barcode, environment_sampled
                 FROM ag.ag_kit_barcodes
                 WHERE environment_sampled IS NOT NULL
                     AND environment_sampled != ''
                     AND barcode IN %s"""
        env_barcodes = self._con.execute_fetchall(sql, [tuple(barcodes)])
        barcodes.extend([b[0] for b in env_barcodes])



        # keep track of which barcodes were seen so we know which weren't
        barcodes_seen = set()
        metadata = {}
        for survey, bc_responses in all_results.items():
            if not bc_responses:
                continue
            # Get the headers for the survey, then union with ones added during
            # pulldown formatting
            headers = set(x[0] for x in
                          self._con.execute_fetchall(header_sql, [survey]))
            headers = headers.union(bc_responses.values()[0])

            # Remove the ebi prohibited columns
            headers = headers.difference(ebi_remove)
            headers = sorted(headers)

            rows = []
            for barcode, shortnames_answers in sorted(bc_responses.items()):
                barcodes_seen.add(barcode)
                oa_hold = [barcode]
                for h in headers:
                    # Take care of retired questions not having an answer
                    answer = shortnames_answers.get(h, 'Not provided')
                    # Convert everything to utf-8 unicode for standardization
                    converted = self._unicode_convert(answer)
                    oa_hold.append(converted)
                rows.append(oa_hold)

            if survey == 1 and blanks:
                # only add blanks to human survey sample data
                for blank in blanks:
                    blanks_copy = copy(blanks_values)
                    blanks_copy['ANONYMIZED_NAME'] = blank
                    blanks_copy['HOST_SUBJECT_ID'] = blank
                    blanks_copy['ASSIGNED_FROM_GEO'] = 'false'
                    rows.append([blank] + [blanks_copy[h] for h in headers])

            qiita_max = 63  # maximum column width in qiita
            headers_ = [h[:qiita_max] for h in headers]
            current_metadata = pd.DataFrame(rows,
                                            columns=['sample_name'] + headers_)
            metadata[survey] = current_metadata.to_csv(sep=b'\t',
                                                       encoding='utf-8',
                                                       index=False)

        if len(env_barcodes) > 0:
            all_results['env'], err = self.format_environmental(env_barcodes)
            for b in all_results['env']:
                barcodes_seen.add(b)
            errors.update(err)

        failures = set(barcodes) - barcodes_seen
        failures = self._explain_pulldown_failures(failures)
        failures.update(errors)
        return metadata, failures

    def _unicode_convert(self, value):
        """Convert given value to unicode string"""
        if isinstance(value, unicode):
            converted = value
        elif isinstance(value, str):
            converted = unicode(value, 'utf-8')
        else:
            converted = unicode(str(value), 'utf-8')
        converted = re.sub(r"\t|\r|\n|\s+", " ", converted)
        return converted

    def check_consent(self, barcodes):
        """Gets barcodes with consent, and failure reasons for ones without

        Parameters
        ----------
        barcodes : list of str
            Barcodes to check for consent

        Returns
        -------
        consented : list of str
            Barcodes with consent
        failures : dict
            Barcodes unable to pull metadata down, in the form
            {barcode: reason, ...}
        """
        sql = """SELECT barcode
                 FROM ag.ag_kit_barcodes
                 LEFT JOIN ag.source_barcodes_surveys USING (barcode)
                 WHERE barcode in %s AND survey_id IS NOT NULL"""
        consented = [x[0] for x in
                     self._con.execute_fetchall(sql, [tuple(barcodes)])]

        failures = set(barcodes).difference(consented)

        return consented, self._explain_pulldown_failures(failures)

    def _explain_pulldown_failures(self, barcodes):
        """Builds failure reason list for barcodes passed

        Parameters
        ----------
        barcodes : list of str
            Barcodes to explain failure for

        Returns
        -------
        dict
            failure reasons in the form {barcode: reason, ...}
        """
        # if empty list passed, don't touch database
        if len(barcodes) == 0:
            return {}

        def update_reason_and_remaining(sql, reason, failures, remaining):
            failures.update(
                {bc[0]: reason for bc in
                 self._con.execute_fetchall(sql, [tuple(remaining)])})
            return remaining.difference(failures)

        fail_reason = {}
        remaining = set(barcodes)
        # TEST ORDER HERE MATTERS! Assumptions made based on filtering of
        # curent_barcodes by previous checks
        # not an AG barcode
        sql = """SELECT barcode
                 FROM ag.ag_kit_barcodes
                 WHERE barcode IN %s
                 UNION
                 SELECT barcode
                 FROM ag.ag_handout_barcodes
                 WHERE barcode IN %s"""
        hold = {x[0] for x in
                self._con.execute_fetchall(
                    sql, [tuple(remaining)] * 2)}
        fail_reason.update({bc: 'Not an AG barcode' for bc in
                            remaining.difference(hold)})
        remaining = hold
        # No more unexplained, so done
        if len(remaining) == 0:
            return fail_reason

        # handout barcode
        sql = """SELECT barcode
                 FROM ag.ag_handout_barcodes
                 WHERE barcode IN %s"""
        remaining = update_reason_and_remaining(
            sql, 'Unassigned handout kit barcode', fail_reason, remaining)
        # No more unexplained, so done
        if len(remaining) == 0:
            return fail_reason

        # withdrawn
        sql = """SELECT barcode
                 FROM ag.ag_kit_barcodes
                 WHERE withdrawn = 'Y' AND barcode in %s"""
        remaining = update_reason_and_remaining(
            sql, 'Withdrawn sample', fail_reason, remaining)
        # No more unexplained, so done
        if len(remaining) == 0:
            return fail_reason

        # sample not logged
        sql = """SELECT barcode
                 FROM ag.ag_kit_barcodes
                 WHERE sample_date IS NULL AND barcode in %s"""
        remaining = update_reason_and_remaining(
            sql, 'Sample not logged', fail_reason, remaining)
        # No more unexplained, so done
        if len(remaining) == 0:
            return fail_reason

        # Sample not consented
        sql = """SELECT barcode
                 FROM ag.ag_kit_barcodes
                 JOIN ag.source_barcodes_surveys USING (barcode)
                 WHERE survey_id IS NULL AND barcode in %s"""
        remaining = update_reason_and_remaining(
            sql, 'Sample logged without consent', fail_reason, remaining)
        # No more unexplained, so done
        if len(remaining) == 0:
            return fail_reason

        # other
        fail_reason.update({bc: 'Unknown reason' for bc in remaining})
        return fail_reason

    def _hash_password(self, password, hashedpw=None):
        """Hashes password

        Parameters
        ----------
        password: str
            Plaintext password
        hashedpw: str, optional
            Previously hashed password for bcrypt to pull salt from. If not
            given, salt generated before hash

        Returns
        -------
        str
            Hashed password

        Notes
        -----
        Relies on bcrypt library to hash passwords, which stores the salt as
        part of the hashed password. Don't need to actually store the salt
        because of this.
        """
        # all the encode/decode as a python 3 workaround for bcrypt
        if hashedpw is None:
            hashedpw = gensalt()
        else:
            hashedpw = hashedpw.encode('utf-8')
        password = password.encode('utf-8')
        output = hashpw(password, hashedpw)
        if isinstance(output, bytes):
            output = output.decode("utf-8")
        return output

    def authenticate_user(self, email, password):
        # see if user exists
        sql = """SELECT EXISTS (SELECT email FROM ag.labadmin_users
                 WHERE email = %s)"""
        exists = self._con.execute_fetchone(sql, [email])[0]

        if not exists:
            raise IncorrectEmailError("Email not valid: %s" % email)

        # pull password out of database
        sql = "SELECT password FROM ag.labadmin_users WHERE email = %s"

        # verify password
        dbpass = self._con.execute_fetchone(sql, [email])
        dbpass = dbpass[0] if dbpass else ''
        hashed = self._hash_password(password, dbpass)

        if hashed == dbpass:
            return True
        else:
            raise IncorrectPasswordError("Password not valid!")

        return False

    def get_unconsented(self):
        """Returns unconsented barcode and person's email

        Returns
        -------
        list of (str, datetime.date, str)
            Unconsented barcodes, as [(barcode, scan_date, email), ...]
        """
        sql = """SELECT DISTINCT barcode, scan_date, email
                 FROM ag.ag_kit_barcodes
                 JOIN barcodes.barcode USING (barcode)
                 JOIN ag.ag_kit USING (ag_kit_id)
                 JOIN ag.ag_login USING (ag_login_id)
                 LEFT JOIN ag.source_barcodes_surveys USING (barcode)
                 WHERE survey_id IS NULL AND scan_date IS NOT NULL
                 ORDER BY barcode"""
        return self._con.execute_fetchall(sql)

    def getAGKitDetails(self, supplied_kit_id):
        sql = """SELECT
                 cast(ag_kit_id AS varchar(100)) AS ag_kit_id, supplied_kit_id,
                 kit_password, swabs_per_kit, kit_verification_code,
                 kit_verified, verification_email_sent
                 FROM ag_kit
                 WHERE supplied_kit_id = %s"""
        res = self._con.execute_fetchone(sql, [supplied_kit_id])
        if res is not None:
            return dict(res)
        else:
            return {}

    def add_barcodes_to_kit(self, ag_kit_id, num_barcodes=1):
        """Attaches barcodes to an existing american gut kit

        Parameters
        ----------
        ag_kit_id : str
            Kit ID to attach barcodes to
        num_barcodes : int, optional
            Number of barcodes to attach. Default 1

        Returns
        -------
        barcodes : list of str
            Barcodes attached to the kit
        """
        barcodes = self.get_unassigned_barcodes(num_barcodes)
        # assign barcodes to projects for the kit
        sql = """SELECT DISTINCT project_id FROM barcodes.project_barcode
                 JOIN ag.ag_kit_barcodes USING (barcode)
                 WHERE ag_kit_id = %s"""
        proj_ids = [x[0] for x in self._con.execute_fetchall(sql, [ag_kit_id])]
        barcode_project_insert = """INSERT INTO project_barcode
                                    (barcode, project_id)
                                    VALUES (%s, %s)"""
        project_inserts = []
        for barcode in barcodes:
            for project in proj_ids:
                project_inserts.append((barcode, project))
        self._con.executemany(barcode_project_insert, project_inserts)

        # Add barcodes to the kit
        sql = """INSERT  INTO ag_kit_barcodes
                (ag_kit_id, barcode, sample_barcode_file)
                VALUES (%s, %s, %s || '.jpg')"""
        barcode_info = [[ag_kit_id, b, b] for b in barcodes]
        self._con.executemany(sql, barcode_info)
        return barcodes

    def create_ag_kits(self, swabs_kits, tag=None, projects=None):
        """ Creates american gut handout kits on the database

        Parameters
        ----------
        swabs_kits : list of tuples
            kits and swab counts, with tuples in the form
            (# of swabs, # of kits with this swab count)
        tag : str, optional
            Tag to add to kit IDs. Default None
        projects : list of str, optional
            Subprojects to attach to, if given. Default None.

        Returns
        -------
        list of namedtuples
            The new kit information, in the form
            [(kit_id, password, verification_code, (barcode, barcode,...)),...]
        """
        # make sure we have enough barcodes
        total_swabs = sum(s * k for s, k in swabs_kits)
        barcodes = self.get_unassigned_barcodes(total_swabs)

        # Assign barcodes to AG and any other subprojects
        if projects is None:
            projects = ["American Gut Project"]
        else:
            if "American Gut Project" not in projects:
                projects.append("American Gut Project")
        self.assign_barcodes(total_swabs, projects)

        kits = []
        kit_barcode_inserts = []
        kit_inserts = []
        start = 0
        KitTuple = namedtuple('AGKit', ['kit_id', 'password',
                              'verification_code', 'barcodes'])
        # build the kits information and the sql insert information
        for num_swabs, num_kits in swabs_kits:
            kit_ids = make_valid_kit_ids(num_kits, self.get_used_kit_ids(),
                                         tag=tag)
            for i in range(num_kits):
                ver_code = make_verification_code()
                password = make_passwd()
                kit_bcs = tuple(barcodes[start:start + num_swabs])
                start += num_swabs
                kits.append(KitTuple(kit_ids[i], password, ver_code, kit_bcs))
                kit_inserts.append((kit_ids[i],
                                    self._hash_password(password),
                                    ver_code, num_swabs))
                for barcode in kit_bcs:
                    kit_barcode_inserts.append((kit_ids[i], barcode, barcode))

        # Insert kits, followed by barcodes attached to the kits
        kit_sql = """INSERT INTO ag_handout_kits
                     (kit_id, password, verification_code, swabs_per_kit)
                     VALUES (%s, %s, %s, %s)"""
        kit_barcode_sql = """INSERT INTO ag_handout_barcodes
                             (kit_id, barcode, sample_barcode_file)
                             VALUES(%s, %s, %s || '.jpg')"""

        self._con.executemany(kit_sql, kit_inserts)
        self._con.executemany(kit_barcode_sql, kit_barcode_inserts)

        return kits

    def get_used_kit_ids(self):
        """Grab in use kit IDs, return set of them
        """
        sql = """SELECT supplied_kit_id FROM ag_kit
                 UNION
                 SELECT kit_id from ag_handout_kits"""

        return set(i[0] for i in self._con.execute_fetchall(sql))

    def create_project(self, name):
        if name.strip() == '':
            raise ValueError("Project name can not be blank!")
        sql = "SELECT EXISTS(SELECT * FROM project WHERE project = %s)"
        exists = self._con.execute_fetchone(sql, [name])[0]
        if exists:
                raise ValueError("Project %s already exists!" %
                                 xhtml_escape(name))

        sql = """INSERT INTO project (project_id, project)
                 SELECT max(project_id)+1, %s FROM project"""
        self._con.execute(sql, [name])

    def get_unassigned_barcodes(self, n=None):
        """Returns unassigned barcodes

        Parameters
        ----------
        n : int, optional
            Number of barcodes to limit to, default returns all unused

        Returns
        -------
        list
            unassigned barcodes

        Raises
        ------
        ValueError
            Not enough unnasigned barcodes for n

        Notes
        -----
        Barcodes are returned in ascending order
        """
        sql = """SELECT DISTINCT barcode FROM barcodes.barcode
                 LEFT JOIN barcodes.project_barcode pb USING (barcode)
                 WHERE pb.barcode IS NULL
                 ORDER BY barcode ASC"""
        barcodes = self._con.execute_fetchall(sql)
        if n is not None:
            if len(barcodes) < n:
                raise ValueError(
                    "Not enough barcodes! %d asked for, %d remaining"
                    % (n, len(barcodes)))
            barcodes = barcodes[:n]
        return [x[0] for x in barcodes]

    def assign_barcodes(self, num_barcodes, projects):
        """Assign a given number of barcodes to projects

        Parameters
        ----------
        num_barcodes : int
            Number of barcodes to assign
        projects : list of str
            Projects to assgn barcodes to

        Returns
        -------
        list of str
            Barcodes assigned to the projects

        Raises
        ------
        ValueError
            One or more projects given don't exist in the database
        """
        # Verify projects given exist
        sql = "SELECT project FROM project"
        existing = {x[0] for x in self._con.execute_fetchall(sql)}
        not_exist = {p for p in projects if p not in existing}
        if not_exist:
            raise ValueError("Project(s) given don't exist in database: %s"
                             % ', '.join(map(xhtml_escape, not_exist)))

        # Get unassigned barcode list and make sure we have enough barcodes
        barcodes = self.get_unassigned_barcodes(num_barcodes)

        # Assign barcodes to the project(s)
        sql = "SELECT project_id from project WHERE project in %s"
        proj_ids = [x[0] for x in
                    self._con.execute_fetchall(sql, [tuple(projects)])]

        barcode_project_insert = """INSERT INTO project_barcode
                                    (barcode, project_id)
                                    VALUES (%s, %s)"""
        project_inserts = []
        for barcode in barcodes:
            for project in proj_ids:
                project_inserts.append((barcode, project))
        self._con.executemany(barcode_project_insert, project_inserts)
        # Set assign date for the barcodes
        sql = """UPDATE barcodes.barcode
                 SET assigned_on = NOW() WHERE barcode IN %s"""
        self._con.execute(sql, [tuple(barcodes)])
        return barcodes

    def create_barcodes(self, num_barcodes):
        """Creates new barcodes

        Parameters
        ----------
        num_barcodes : int
            Number of barcodes to create

        Returns
        -------
        list
            New barcodes created
        """

        # Get newest barcode as an integer
        sql = "SELECT max(barcode::integer) from barcode"
        newest = self._con.execute_fetchone(sql)[0]

        # create new barcodes by padding integers with zeros
        barcodes = ['%09d' % b for b in range(newest+1, newest+1+num_barcodes)]

        barcode_insert = """INSERT INTO barcode (barcode, obsolete)
                            VALUES (%s, 'N')"""
        self._con.executemany(barcode_insert, [[b] for b in barcodes])
        return barcodes

    def get_barcodes_for_projects(self, projects, limit=None):
        """Gets barcode information for barcodes belonging to projects

        Parameters
        ----------
        projects : list of str
            Projects to get barcodes for (if multiple given, intersection of
            barcodes in each project is returned)
        limit : int, optional
            Number of barcodes to return, starting with most recent
            (defult all)

        Returns
        -------
        list of dict
            each barcode with information
        """
        select_sql = """SELECT barcode, create_date_time, sample_postmark_date,
                        scan_date,status,sequencing_status,biomass_remaining,
                        obsolete,array_agg(project) AS projects
                        FROM barcode
                        JOIN project_barcode USING (barcode)
                        JOIN project USING (project_id)
                        WHERE project IN %s GROUP BY barcode
                        ORDER BY barcode DESC"""
        sql_args = [tuple(projects)]
        if limit is not None:
            select_sql += " LIMIT %s"
            sql_args.append(limit)
        return self._con.execute_fetchall(select_sql, sql_args)

    def add_external_survey(self, survey, description, url):
        """Adds a new external survey to the database

        Parameters
        ----------
        survey : str
            Name of the external survey
        description : str
            Short description of what the survey is about
        url : str
            URL for the external survey

        Raises
        ------
        ValueError
            survey already exists in DB
        """
        sql = """SELECT EXISTS(
                    SELECT external_survey
                    FROM ag.external_survey_sources
                    WHERE external_survey = %s)"""
        if self._con.execute_fetchone(sql, [survey])[0]:
            raise ValueError("Survey '%s' already exists" % survey)

        sql = """INSERT INTO ag.external_survey_sources
                 (external_survey, external_survey_description,
                  external_survey_url)
                 VALUES (%s, %s, %s)
                 RETURNING external_survey_id"""
        return self._con.execute_fetchone(sql, [survey, description, url])[0]

    def list_ag_surveys(self, selected=None):
        """Returns the list of american gut survey names.

        Parameters
        ----------
        selected : list of int
            The returned list's third element indicates if a survey has been
            "chosen". If selected is None, all surveys will be "chosen",
            otherwise only surveys whose ID is in selected are "chosen".

        Returns
        -------
        list of (int, str, bool)
            first element is the group_order number
            second element is the group name
            third element is the information if user selected this survey for
                  pulldown. All surveys are selected if selected is None.
        """
        sql = """SELECT group_order, american
                FROM ag.survey_group
                WHERE group_order < 0"""
        return [(id_, name, (selected is None) or (id_ in selected))
                for [id_, name] in self._con.execute_fetchall(sql)]

    def list_external_surveys(self):
        """Returns list of external survey names

        Returns
        -------
        list of str
            Third party survey names
        """
        sql = """SELECT external_survey
                 FROM ag.external_survey_sources"""
        return [x[0] for x in self._con.execute_fetchall(sql)]

    def store_external_survey(self, in_file, ext_survey, pulldown_date=None,
                              separator="\t", survey_id_col="survey_id",
                              trim=None):
        """Stores third party survey answers in the database

        Parameters
        ----------
        in_file : open file or StringIO
            File with survey spreadsheet
        external_survey_urlsurvey : str
            What third party survey this belongs to
        pulldown_date : datetime object, optional
            When the data was pulled from the external source, default now()
        separator : str, optional
            What separator is used, default tab
        survey_id_col : str
            What column header holds the associated user AG survey id
            Default 'survey_id'
        trim : str
            Regex to trim the survey id column, using re.sub(trim, '', sid)
            Default None

        Returns
        -------
        count : int
            Number of rows inserted

        Raises
        ------
        ValueError
            Survey passed is not found
        """
        # Get the external survey ID
        sql = """SELECT external_survey_id
                 FROM external_survey_sources
                 WHERE external_survey = %s"""
        external_id = self._con.execute_fetchone(sql, [ext_survey])
        if not external_id:
            raise ValueError("Unknown external survey: %s" % ext_survey)
        external_id = external_id[0]
        if pulldown_date is None:
            pulldown_date = datetime.now()

        # Load file data into insertable json format
        data = pd.read_csv(in_file, sep=',', dtype=str, na_filter=False)
        data.set_index(survey_id_col, inplace=True)

        if ext_survey == 'Vioscreen':
            # in the provided vioscreen files, the headers Calcium and calcium
            # are present. The subsequent consumer is not assured to handle
            # this gracefully if case is disregarded (which it is for sample
            # information). Rewriting "calcium" to "calcium.1" in accordance
            # with how this header was specified in the manuscript submission.
            data = data.rename(columns={'calcium': 'calcium.1'})

        inserts = []
        for sid, row in data.iterrows():
            if trim is not None:
                sid = re.sub(trim, '', sid)
            inserts.append([sid, external_id, pulldown_date,
                            json.dumps(row.to_dict())])

        # insert into the database
        sql = """INSERT INTO ag.external_survey_answers
                 (survey_id, external_survey_id, pulldown_date, answers)
                 VALUES (%s, %s, %s, %s)"""
        self._con.executemany(sql, inserts)
        return len(inserts)

    def get_external_survey(self, survey, survey_ids, pulldown_date=None):
        """Get the answers to a survey for given survey IDs

        Parameters
        ----------
        survey : str
            Survey to retrieve answers for
        survey_ids : list of str
            AG survey ids to retrieve answers for
        pulldown_date : datetime object, optional
            Specific pulldown date to limit answers to, default None

        Returns
        -------
        dict of dicts
            Answers to the survey keyed to the given survey IDs, in the form
            {survey_id: {header1: answer, header2: answer, ...}, ...}

        Notes
        -----
        If there are multiple pulldowns for a given survey_id, the newest one
        will be returned.
        """
        # Do pulldown of ids and answers, ordered so newest comes out last
        # This allows you to not specify pulldown date and still get newest
        # answers for the survey
        sql = """SELECT survey_id, answers FROM
                 (SELECT * FROM ag.external_survey_answers
                 JOIN ag.external_survey_sources USING (external_survey_id)
                 WHERE external_survey = %s AND survey_id IN %s{0}
                 ORDER BY pulldown_date ASC) AS A"""
        sql_args = [survey, tuple(survey_ids)]
        format_str = ""
        if pulldown_date is not None:
            format_str = " AND pulldown_date = %s "
            sql_args.append(pulldown_date)

        info = self._con.execute_fetchall(sql.format(format_str), sql_args)
        if info:
            return {s: a for s, a in info}
        else:
            return {}

    def addAGLogin(self, email, name, address, city, state, zip_, country):
        clean_email = email.strip().lower()
        sql = "select ag_login_id from ag_login WHERE LOWER(email) = %s"
        ag_login_id = self._con.execute_fetchone(sql, [clean_email])
        if not ag_login_id:
            # create the login
            sql = ("INSERT INTO ag_login (email, name, address, city, state, "
                   "zip, country) VALUES (%s, %s, %s, %s, %s, %s, %s) "
                   "RETURNING ag_login_id")
            ag_login_id = self._con.execute_fetchone(
                sql, [clean_email, name, address, city, state, zip_, country])
        return ag_login_id[0]

    def updateAGLogin(self, ag_login_id, email, name, address, city, state,
                      zipcode, country):
        sql = """UPDATE  ag_login
                SET email = %s, name = %s, address = %s, city = %s, state = %s,
                    zip = %s, country = %s
                WHERE ag_login_id = %s"""
        self._con.execute(sql, [email.strip().lower(), name,
                                address, city, state, zipcode, country,
                                ag_login_id])

    def updateAGBarcode(self, barcode, ag_kit_id, site_sampled,
                        environment_sampled, sample_date, sample_time,
                        participant_name, notes, refunded, withdrawn):

        # convert empty strings to None for DB consistency
        site_sampled = site_sampled or None
        environment_sampled = environment_sampled or None
        sample_date = sample_date or None
        sample_time = sample_time or None
        notes = notes or None

        sql = """UPDATE  ag_kit_barcodes
                 SET ag_kit_id = %s,
                     site_sampled = %s,
                     environment_sampled = %s,
                     sample_date = %s,
                     sample_time = %s,
                     notes = %s,
                     refunded = %s,
                     withdrawn = %s
                 WHERE barcode = %s"""
        self._con.execute(sql, [ag_kit_id, site_sampled, environment_sampled,
                                sample_date, sample_time,
                                notes, refunded, withdrawn,
                                barcode])

        # update assignment of barcode to source
        # delete existing assignments for the given barcode
        sql_remove = """DELETE FROM ag.source_barcodes_surveys
                        WHERE barcode = %s"""
        self._con.execute(sql_remove, [barcode])

        if participant_name is not None:
            # determine surveys for source (= ag_login_id + participant_name)
            # with this barcode
            sql = """SELECT survey_id
                     FROM ag.ag_kit_barcodes
                     LEFT JOIN ag.ag_kit USING (ag_kit_id)
                     LEFT JOIN ag.ag_login_surveys USING (ag_login_id)
                     WHERE barcode = %s AND participant_name = %s"""
            survey_ids = self._con.execute_fetchall(sql, [barcode,
                                                          participant_name])
            if survey_ids is not None:
                survey_ids = [x[0] for x in survey_ids]
                # create a new association between source and barcode,
                # i.e. assign barcode to source
                sql_insert = """INSERT INTO ag.source_barcodes_surveys
                                (survey_id, barcode) VALUES (%s, %s)"""
                for survey_id in survey_ids:
                    self._con.execute(sql_insert, [survey_id, barcode])

    def AGGetBarcodeMetadata(self, barcode):
        results = self._con.execute_proc_return_cursor(
            'ag_get_barcode_metadata', [barcode])
        rows = results.fetchall()
        results.close()

        return [dict(row) for row in rows]

    def AGGetBarcodeMetadataAnimal(self, barcode):
        results = self._con.execute_proc_return_cursor(
            'ag_get_barcode_md_animal', [barcode])
        rows = results.fetchall()
        results.close()

        return [dict(row) for row in rows]

    def get_geocode_zipcode(self, zipcode, country):
        """Adds geocode information to zipcode table if needed and return info

        Parameters
        ----------
        zipcode : str
            Zipcode to geocode
        country : str, optional
            Country zipcode belongs in. Default infer from zipcode. Useful
            for countries with zipcode formated like USA

        Returns
        -------
        info : NamedTuple
            Location namedtuple in form
            Location('zip', 'lat', 'long', 'elev', 'city', 'state', 'country')

        Notes
        -----
        If the tuple contains nothing but the zipcode and None for all other
        fields, no geocode was found. Zipcode/country combination added as
        'cannot_geocode'
        """
        # Catch sending None or empty string for these
        if not zipcode or not country:
            return Location(zipcode, None, None, None,
                            None, None, None, country)
        info = geocode('%s %s' % (zipcode, country))
        cannot_geocode = False
        # Clean the zipcode so it is same case and setup, since international
        # people can enter lowercased zipcodes or missing spaces, and google
        # does not give back 9 digit zipcodes for USA, only 6.
        # take care of utf-8 chars, thus en- and de-code accordingly
        clean_postcode = None
        if info.postcode is not None:
            clean_postcode = str(info.postcode.encode('utf-8')).lower()\
                .decode('utf-8').replace(' ', '')
        clean_zipcode = str(zipcode.encode('utf-8')).lower().decode('utf-8')\
            .replace(' ', '').split('-')[0]
        help_pc_identity = False
        if clean_postcode is not None:
            help_pc_identity = clean_postcode.startswith(clean_zipcode)
        if not info.lat:
            cannot_geocode = True
        # Use startswith because UK zipcodes can be 2, 3, or 6 characters
        elif (info.country != country or
              help_pc_identity):
            # countries and zipcodes dont match, so blank out info
            info = Location(zipcode, None, None, None,
                            None, None, None, country)
            cannot_geocode = True
        sql = """INSERT INTO ag.zipcodes
                    (zipcode, latitude, longitude, elevation, city,
                     state, country, cannot_geocode)
                 VALUES (%s,%s,%s,%s,%s,%s,%s, %s)"""
        self._con.execute(sql, [zipcode, info.lat, info.long, info.elev,
                                info.city, info.state, country,
                                cannot_geocode])
        return info

    def addGeocodingInfo(self, limit=None, retry=False):
        """Adds latitude, longitude, and elevation to ag_login_table

        Uses the city, state, zip, and country from the database to retrieve
        lat, long, and elevation from the google maps API.

        If any of that information cannot be retrieved, then cannot_geocode
        is set to 'y' in the ag_login table, and it will not be tried again
        on subsequent calls to this function.  Pass retry=True to retry all
        (or maximum of limit) previously failed geocodings.
        """

        # clear previous geocoding attempts if retry is True
        if retry:
            sql = """UPDATE  ag_login
                     SET latitude = %s,
                         longitude = %s,
                         elevation = %s,
                         cannot_geocode = %s
                     WHERE ag_login_id IN (
                        SELECT ag_login_id FROM ag_login
                        WHERE cannot_geocode = 'y')"""
            self._con.execute(sql)

        # get logins that have not been geocoded yet
        sql = """SELECT city, state, zip, country,
                        cast(ag_login_id as varchar(100))
                 FROM ag_login
                 WHERE elevation is NULL AND cannot_geocode is NULL"""
        logins = self._con.execute_fetchall(sql)

        row_counter = 0
        sql_args = []
        for city, state, zipcode, country, ag_login_id in logins:
            row_counter += 1
            if limit is not None and row_counter > limit:
                break

            # Attempt to geocode
            address = '%s %s %s %s' % (
                city.decode('utf-8'), state.decode('utf-8'),
                zipcode.decode('utf-8'), country.decode('utf-8'))
            try:
                info = geocode(address)
                # empty string to indicate geocode was successful
                sql_args.append([info.lat, info.long, info.elev,
                                 '', ag_login_id])
            except GoogleAPILimitExceeded:
                # limit exceeded so no use trying to keep geocoding
                break
            except:
                # Catch ANY other error and set to could not geocode
                sql_args.append([None, None, None, 'y', ag_login_id])

        sql = """UPDATE  ag_login
                 SET latitude = %s,
                     longitude = %s,
                     elevation = %s,
                 cannot_geocode = %s
                 WHERE ag_login_id = %s"""
        self._con.executemany(sql, sql_args)

    def getGeocodeStats(self):
        stat_queries = [
            ("Total Rows",
             "select count(*) from ag_login"),
            ("Cannot Geocode",
             "select count(*) from ag_login where cannot_geocode = 'y'"),
            ("Null Latitude Field",
             "select count(*) from ag_login where latitude is null"),
            ("Null Elevation Field",
             "select count(*) from ag_login where elevation is null")
        ]
        results = []
        for name, sql in stat_queries:
            total = self._con.execute_fetchone(sql)[0]
            results.append((name, total))
        return results

    def getAGStats(self):
        # returned tuple consists of:
        # site_sampled, sample_date, sample_time, participant_name,
        # environment_sampled, notes
        stats_list = [
            ('Total handout kits',
             'SELECT count(*) FROM ag.ag_handout_kits'),
            ('Total handout barcodes',
             'SELECT count(*) FROM ag.ag_handout_barcodes'),
            ('Total consented participants',
             'SELECT count(*) FROM ag.ag_consent'),
            ('Total registered kits',
             'SELECT count(*) FROM ag.ag_kit'),
            ('Total registered barcodes',
             'SELECT count(*) FROM ag.ag_kit_barcodes'),
            ('Total barcodes with results',
             """SELECT count(*) FROM ag.ag_kit_barcodes
             WHERE results_ready='Y'"""),
            ('Average age of participants',
             """SELECT AVG(AGE((yr.response || '-' ||
                CASE mo.response
                    WHEN 'January' THEN '1'
                    WHEN 'February' THEN '2'
                    WHEN 'March' THEN '3'
                    WHEN 'April' THEN '4'
                    WHEN 'May' THEN '5'
                    WHEN 'June' THEN '6'
                    WHEN 'July' THEN '7'
                    WHEN 'August' THEN '8'
                    WHEN 'September' THEN '9'
                    WHEN 'October' THEN '10'
                    WHEN 'November' THEN '11'
                    WHEN 'December' THEN '12'
                  END || '-1')::date
                )) FROM
              (SELECT response, survey_id
               FROM ag.survey_answers
               WHERE survey_question_id = 112) AS yr
              JOIN
              (SELECT response, survey_id
               FROM ag.survey_answers
               WHERE survey_question_id = 111) AS mo USING (survey_id)
               WHERE yr.response != 'Unspecified'
               AND mo.response != 'Unspecified'"""),
            ('Total male participants',
             """SELECT count(*) FROM ag.survey_answers
                WHERE survey_question_id=107 AND response='Male'"""),
            ('Total female participants',
             """SELECT count(*) FROM ag.survey_answers
                WHERE survey_question_id=107 AND response='Female'""")
            ]
        stats = []
        for label, sql in stats_list:
            res = self._con.execute_fetchone(sql)[0]
            if type(res) == timedelta:
                res = str(res.days/365) + " years"
            stats.append((label, res))
        return stats

    def updateAKB(self, barcode, moldy, overloaded, other, other_text,
                  date_of_last_email):
        """ Update ag_kit_barcodes table.
        """
        sql_args = [moldy, overloaded, other, other_text]
        update_date = ''
        if date_of_last_email:
            update_date = ', date_of_last_email = %s'
            sql_args.append(date_of_last_email)
        sql_args.append(barcode)

        sql = """UPDATE  ag_kit_barcodes
                 SET moldy = %s, overloaded = %s, other = %s,
                     other_text = %s{}
                 WHERE barcode = %s""".format(update_date)
        self._con.execute(sql, sql_args)

    def updateBarcodeStatus(self, status, postmark, scan_date, barcode,
                            biomass_remaining, sequencing_status, obsolete):
        """ Updates a barcode's status
        """
        sql = """UPDATE  barcode
                 SET status = %s,
                     sample_postmark_date = %s,
                     scan_date = %s,
                     biomass_remaining = %s,
                     sequencing_status = %s,
                     obsolete = %s
                 WHERE barcode = %s"""
        self._con.execute(sql, [status, postmark, scan_date, biomass_remaining,
                                sequencing_status, obsolete, barcode])

    def get_barcode_survey(self, barcode):
        """Return survey ID attached to barcode"""
        sql = """SELECT DISTINCT ags.survey_id
                 FROM ag.ag_kit_barcodes
                 JOIN ag.source_barcodes_surveys USING (barcode)
                 JOIN ag.survey_answers USING (survey_id)
                 JOIN ag.group_questions gq USING (survey_question_id)
                 JOIN ag.surveys ags USING (survey_group)
                 WHERE barcode = %s"""
        res = self._con.execute_fetchone(sql, [barcode])
        return res[0] if res else None

    def search_participant_info(self, term):
        sql = """SELECT cast(ag_login_id as varchar(100)) as ag_login_id
                 FROM ag_login al
                 WHERE lower(email) like %s or lower(name) like
                 %s or lower(address) like %s"""
        liketerm = '%%' + term.lower() + '%%'
        results = self._con.execute_fetchall(sql,
                                             [liketerm, liketerm, liketerm])
        return [x[0] for x in results]

    def search_kits(self, term):
        sql = """SELECT cast(ag_login_id as varchar(100)) as ag_login_id
                 FROM ag_kit
                 WHERE lower(supplied_kit_id) like %s or
                 lower(kit_password) like %s or
                 lower(kit_verification_code) = %s or
                 cast(ag_kit_id as varchar(100)) like %s"""
        liketerm = '%%' + term.lower() + '%%'
        results = self._con.execute_fetchall(sql,
                                             [liketerm, liketerm, liketerm,
                                              liketerm])
        return [x[0] for x in results]

    def search_barcodes(self, term):
        sql = """SELECT DISTINCT
                    cast(ag_login_id as varchar(100)) as ag_login_id
                 FROM ag.ag_kit_barcodes
                 JOIN ag.ag_kit USING (ag_kit_id)
                 LEFT JOIN ag.ag_login_surveys USING (ag_login_id)
                 WHERE barcode like %s or lower(participant_name) like
                 %s or lower(notes) like %s"""
        liketerm = '%%' + term.decode('utf-8').lower() + '%%'
        results = self._con.execute_fetchall(sql,
                                             [liketerm, liketerm, liketerm])
        return [x[0] for x in results]

    def get_kit_info_by_login(self, ag_login_id):
        sql = """SELECT cast(ag_kit_id as varchar(100)) as ag_kit_id,
                        cast(ag_login_id as varchar(100)) as ag_login_id,
                        supplied_kit_id, kit_password, swabs_per_kit,
                        kit_verification_code, kit_verified
                 FROM ag_kit
                 WHERE ag_login_id = %s"""
        info = self._con.execute_fetchdict(sql, [ag_login_id])
        return info if info else []

    def search_handout_kits(self, term):
        sql = """SELECT kit_id, password, barcode, verification_code
                 FROM ag.ag_handout_kits
                 JOIN (SELECT kit_id, barcode, sample_barcode_file
                    FROM ag.ag_handout_barcodes
                    GROUP BY kit_id, barcode) AS hb USING (kit_id)
                 WHERE kit_id LIKE %s or barcode LIKE %s"""
        liketerm = '%%' + term + '%%'
        return self._con.execute_fetchdict(sql, [liketerm, liketerm])

    def get_login_by_email(self, email):
        sql = """SELECT name, address, city, state, zip, country, ag_login_id
                 FROM ag_login WHERE email = %s"""
        row = self._con.execute_fetchone(sql, [email])

        login = {}
        if row:
            login = dict(row)
            login['email'] = email

        return login

    def get_login_info(self, ag_login_id):
        sql = """SELECT  ag_login_id, email, name, address, city, state, zip,
                         country
                 FROM    ag_login
                 WHERE   ag_login_id = %s"""
        return self._con.execute_fetchdict(sql, [ag_login_id])

    def getAGBarcodeDetails(self, barcode):
        sql = """SELECT DISTINCT email,
                    cast(ag_kit_barcode_id as varchar(100)),
                    cast(ag_kit_id as varchar(100)), barcode,  site_sampled,
                    environment_sampled, sample_date, sample_time,
                    participant_name, notes, refunded, withdrawn, moldy, other,
                    other_text, date_of_last_email ,overloaded, name, status,
                    deposited
                 FROM ag.ag_kit_barcodes akb
                 JOIN ag.ag_kit USING(ag_kit_id)
                 JOIN ag.ag_login USING (ag_login_id)
                 LEFT JOIN ag.source_barcodes_surveys USING (barcode)
                 LEFT JOIN ag.ag_login_surveys USING (survey_id)
                 JOIN barcode USING (barcode)
                 WHERE barcode = %s"""

        results = self._con.execute_fetchone(sql, [barcode])
        if not results:
            return {}
        else:
            return dict(results)

    def get_barcode_info_by_kit_id(self, ag_kit_id):
        sql = """SELECT DISTINCT cast(ag_kit_barcode_id as varchar(100)) as
                         ag_kit_barcode_id, cast(ag_kit_id as varchar(100)) as
                         ag_kit_id, barcode, sample_date, sample_time,
                         site_sampled, environment_sampled, participant_name,
                         notes, results_ready, withdrawn, refunded
                 FROM ag.ag_kit_barcodes
                 LEFT JOIN ag.source_barcodes_surveys USING (barcode)
                 LEFT JOIN ag.ag_login_surveys USING (survey_id)
                 WHERE  ag_kit_id = %s"""

        results = [dict(row) for row in
                   self._con.execute_fetchall(sql, [ag_kit_id])]
        return results

    def get_barcodes_with_results(self):
        """Returns list of all barcodes with results ready (PDFs available)

        Returns
        -------
        list of str
            All barcodes with result PDFs available

        Raises
        ------
        IOError
            PDF directory does not exist
        """
        path = join(self.config.base_data_dir, 'pdfs')
        if not isdir(path):
            raise IOError('Unknown folder %s' % abspath(path))
        # Get the list of barcodes from the PDF names
        files = next(walk(path))[2]
        return [splitext(f)[0] for f in files if f.endswith('.pdf')]

    def mark_results_ready(self, barcodes, debug=False):
        """Marks the list of barcodes as ready in the databse and sends email

        Parameters
        ----------
        barcodes: iterable of str
            Barcodes to mark as having results ready

        Notes
        -----
        This function automatically sends emails out to only newly set results
        ready barcodes. This means you can pass in a list of all barcodes from
        all rounds that have results ready, and the function will filter for
        new results automatically.
        """
        debug = {}
        ready_sql = """UPDATE ag.ag_kit_barcodes
                       SET results_ready = 'Y'
                       WHERE barcode IN %s
                       AND (results_ready != 'Y' OR results_ready IS NULL)
                       RETURNING barcode"""
        new_bcs = tuple(x[0] for x in
                        self._con.execute_fetchall(
                            ready_sql, [tuple(barcodes)]))
        debug['new_bcs'] = new_bcs
        if len(new_bcs) == 0:
            # No new barcodes, so no emails to send
            return

        bc_sql = """UPDATE ag.ag_kit_barcodes
                 SET date_of_last_email = '{0}'
                 WHERE barcode IN %s""".format(datetime.now())
        subject = "Your American/British Gut results are ready"
        message = (
            "Good afternoon American & British Gut participants!\n\n"
            "We are pleased to let you know that your results are now "
            "available. You may access them by signing onto "
            "microbio.me/americangut or microbio.me/britishgut. If you have "
            "forgotten your login credentials, you may retrieve them using "
            "the \"Forgot kit ID/password\" functions.\n\n"
            "We thank you for being a part of the project. While we emphasize "
            "getting results back to you, the participant, we and the broader "
            "American/British Gut scientific collaborative network are "
            "extremely excited about the population-scale microbiome "
            "observations that are for the first time becoming possible thanks"
            " to you and the other participants!\n\n"
            "Regards,\n"
            "The American Gut Team\n")
        barcode_info = self.get_ag_barcode_details(new_bcs)
        # Make sure email only sent once if multiple barcodes with same email
        seen_emails = set(i['email'] for bc, i in viewitems(barcode_info))
        mail = send_email(message, subject, bcc=list(seen_emails), debug=debug)
        debug['mail'] = mail
        self._con.execute(bc_sql, [new_bcs])
        if debug:
            return debug

    def getHumanParticipants(self, ag_login_id):
        # get people from new survey setup
        sql = """SELECT DISTINCT participant_name from ag.ag_login_surveys
                 LEFT JOIN ag.survey_answers USING (survey_id)
                 JOIN ag.group_questions gq USING (survey_question_id)
                 JOIN ag.surveys ags USING (survey_group)
                 WHERE ag_login_id = %s AND ags.survey_id = %s"""
        results = self._con.execute_fetchall(sql, [ag_login_id, 1])
        return [row[0] for row in results]

    def getAGKitsByLogin(self):
        sql = """SELECT  lower(al.email) as email, supplied_kit_id,
                 cast(ag_kit_id as varchar(100)) as ag_kit_id
                 FROM ag_login al
                 INNER JOIN ag_kit USING (ag_login_id)
                 ORDER BY lower(email), supplied_kit_id"""
        rows = self._con.execute_fetchall(sql)
        return [dict(row) for row in rows]

    def getAnimalParticipants(self, ag_login_id):
        sql = """SELECT DISTINCT participant_name from ag.ag_login_surveys
                 JOIN ag.survey_answers USING (survey_id)
                 JOIN ag.group_questions gq USING (survey_question_id)
                 JOIN ag.surveys ags USING (survey_group)
                 WHERE ag_login_id = %s AND ags.survey_id = %s"""
        return [row[0] for row in self._con.execute_fetchall(
            sql, [ag_login_id, 2])]

    def ag_new_survey_exists(self, barcode):
        """
        Returns metadata for an american gut barcode in the new database
        tables
        """
        sql = "SELECT EXISTS(SELECT * from ag_kit_barcodes WHERE barcode = %s)"
        return self._con.execute_fetchone(sql, [barcode])[0]

    def get_plate_for_barcode(self, barcode):
        """
        Gets the sequencing plates a barcode is on
        """
        sql = """SELECT p.plate, p.sequence_date
                 FROM plate p
                 INNER JOIN plate_barcode pb
                 ON pb.plate_id = p.plate_id \
                 WHERE pb.barcode = %s"""

        return [dict(row) for row in
                self._con.execute_fetchall(sql, [barcode])]

    def getBarcodeProjType(self, barcode):
        """ Get the project type of the barcode.
            Return a tuple of projects and parent project.
        """
        sql = """SELECT project from barcodes.project
                 JOIN barcodes.project_barcode USING (project_id)
                 where barcode = %s"""
        results = [self._unicode_convert(x[0])
                   for x in self._con.execute_fetchall(sql, [barcode])]
        if u'American Gut Project' in results:
            parent_project = u'American Gut'
            results.remove(u'American Gut Project')
            projects = u', '.join(results)
        else:
            projects = u', '.join(results)
            parent_project = projects
        return (projects, parent_project)

    def setBarcodeProjects(self, barcode,
                           add_projects=None,
                           rem_projects=None):
        """Sets the projects barcode is associated with

        Parameters
        ----------
        barcode : str
            Barcode to update
        add_projects : list of str, optional
            List of projects from projects table to add project to
        rem_projects : list of str, optional
            List of projects from projects table to remove barcode from
        """
        if add_projects:
            sql = """INSERT INTO barcodes.project_barcode
                      SELECT project_id, %s FROM (
                        SELECT project_id from barcodes.project
                        WHERE project in %s)
                     AS P"""

            self._con.execute(sql, [barcode, tuple(add_projects)])
        if rem_projects:
            sql = """DELETE FROM barcodes.project_barcode
                     WHERE barcode = %s AND project_id IN (
                       SELECT project_id
                       FROM barcodes.project WHERE project IN %s)"""
            self._con.execute(sql, [barcode, tuple(rem_projects)])

    def getProjectNames(self):
        """Returns a list of project names
        """
        sql = """SELECT project FROM project"""
        return [self._unicode_convert(x[0])
                for x in self._con.execute_fetchall(sql)]

    def set_deposited_ebi(self):
        """Updates barcode deposited status by checking EBI"""
        accession = 'ERP012803'
        samples = fetch_url(
            'http://www.ebi.ac.uk/ena/data/warehouse/filereport?accession='
            '%s&result=read_run&fields=sample_alias' % accession)
        # Clean EBI formatted sample names to just the barcodes
        # stripped of any appended letters for barcodes run multiple times
        barcodes = tuple(s.strip().split('.')[1][:9]
                         for s in samples if len(s.split('.')) == 2)

        sql = """UPDATE ag.ag_kit_barcodes
                 SET deposited = TRUE
                 WHERE barcode IN %s"""
        self._con.execute(sql, [barcodes])

    def _clear_table(self, table, schema):
        """Test helper to wipe out a database table"""
        self._con.execute('DELETE FROM %s.%s' % (schema, table))

    def _revert_ready(self, barcodes):
        """Test helper to revert barcodes set as ready"""
        sql = """UPDATE ag.ag_kit_barcodes
                 SET results_ready = NULL
                 WHERE barcode IN %s"""
        self._con.execute(sql, [tuple(barcodes)])

    def ut_remove_external_survey(self, name, description, url):
        """ Remove an external survey from DB.
        For unit testing only!

        Parameters
        ----------
        name : str
            Name of the external survey
        description : str
            Description of the external survey
        url : str
            URL of the external survey

        Returns
        -------
        Nothing
        """
        sql = """DELETE FROM ag.external_survey_sources
                 WHERE external_survey = %s
                 AND external_survey_description = %s
                 AND external_survey_url = %s"""
        try:
            self._con.execute(sql, [name, description, url])
        except ValueError as e:
            raise ValueError("Something went wrong: " + str(e))

    def ut_get_participant_names_from_ag_login_id(self, ag_login_id):
        """ Returns all participant_name(s) for a given ag_login_id.
        For unit testing only!

        Parameters
        ----------
        ag_login_id : str
            Existing ag_login_id.

        Returns
        -------
        [[str]]
            Example: ["Name - z\xc3\x96DOZ8(Z~'",
                      "Name - z\xc3\x96DOZ8(Z~'",
                      'Name - QpeY\xc3\xb8u#0\xc3\xa5<',
                      'Name - S)#@G]xOdL',
                      'Name - Y5"^&sGQiW',
                      'Name - L\xc3\xa7+c\r\xc3\xa5?\r\xc2\xbf!',
                      'Name - (~|w:S\xc3\x85#L\xc3\x84']

        Raises
        ------
        ValueError
            If ag_login_id is not in DB.
        """
        sql = """SELECT participant_name
                 FROM ag.ag_login_surveys
                 WHERE ag_login_id = %s"""
        info = self._con.execute_fetchall(sql, [ag_login_id])
        if not info:
            raise ValueError('ag_login_id not in database: %s' %
                             ag_login_id)
        return [n[0] for n in info]

    def ut_get_supplied_kit_id(self, ag_login_id):
        """ Returns supplied_kit_id for a given ag_login_id.
        For unit testing only!

        Parameters
        ----------
        ag_login_id : str
            Existing ag_login_id.

        Returns
        -------
        str
            The supplied_kit_id for the given ag_login_id.
            Example: 'DokBF'

        Raises
        ------
        ValueError
            If ag_login_id is not in DB.
        """
        sql = """SELECT supplied_kit_id
                 FROM ag.ag_kit
                 WHERE ag_login_id = %s"""
        info = self._con.execute_fetchall(sql, [ag_login_id])
        if not info:
            raise ValueError('ag_login_id not in database: %s' %
                             ag_login_id)
        return info[0][0]

    def ut_get_arbitrary_handout_barcode(self, is_AGP=True):
        """ Returns an arbitrarily chosen barcode that is handed out.
        For unit testing only!

        Parameters
        ----------
        is_AGP : Boolean
            If True, only barcodes for American Gut Project are returned,
            otherwise barcode is from some arbitrary project.

        Returns
        -------
        str: barcode
            Example: '000001000'

        Raises
        ------
        ValueError
            If no handed out barcode can be found in the DB.

        Notes
        -----
        If is_AGP is True, this barcode belongs to the AGP, otherwise to any
        project.

        """
        sql = """SELECT barcode FROM ag.ag_handout_barcodes
                 JOIN barcodes.project_barcode USING (barcode)"""
        if is_AGP:
            sql += """ WHERE project_id=1"""
        sql += """ LIMIT 1"""
        info = self._con.execute_fetchone(sql, [])
        if not info:
            raise ValueError('No handout barcodes found.')
        return info[0]

    def ut_get_arbitrary_non_ag_barcode(self):
        """Returns an artibtrarily chosen non-AG barcode
        For unit testing only!

        Returns
        -------
        str: barcode
            Example: '000001000'

        Raises
        ------
        ValueError
            If no non-AG barcode can be found in the DB.
        """
        sql = """SELECT barcode FROM barcodes.project_barcode
                    WHERE project_id != 1 AND barcode != '000000001'
                 EXCEPT
                 SELECT barcode FROM barcodes.project_barcode
                    WHERE project_id = 1;"""
        info = self._con.execute_fetchall(sql)
        if not info:
            raise ValueError('No non-AG barcodes found')
        return info[0][0]

    def ut_get_arbitrary_unassigned_barcode(self):
        """ Returns an arbitrarily chosen unassigned barcode.
        For unit testing only!

        Returns
        -------
        str: barcode
            Example: '000001000'

        Raises
        ------
        ValueError
            If no unassigned barcode can be found in the DB.
        """
        sql = """SELECT barcode FROM ag.ag_handout_barcodes LIMIT 1"""
        info = self._con.execute_fetchone(sql, [])
        if not info:
            raise ValueError('No unassigned barcodes found.')
        return info[0]

    def ut_remove_project(self, project_name):
        """ Deletes a project.
        For unit testing only!

        Parameters
        ----------
        project_name : str
            The name of the project that should be deleted.

        Raises
        ------
        ValueError
            If project could not be deleted, e.g. because barcodes are still
            associated to this project."""
        sql = """DELETE FROM barcodes.project WHERE project=%s"""
        try:
            self._con.execute(sql, [project_name])
        except:
            raise ValueError("Unable to delete project %s" % project_name)
