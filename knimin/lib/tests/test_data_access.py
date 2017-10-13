from unittest import TestCase, main
from os.path import join, dirname, realpath
from six import StringIO
import datetime

import numpy as np
from hashlib import sha512
import pandas.util.testing as pdt
import pandas as pd

from knimin import db
from knimin.lib.constants import ebi_remove


class TestDataAccess(TestCase):
    ext_survey_fp = join(dirname(realpath(__file__)), '..', '..', 'tests',
                         'data', 'external_survey_data.csv')
    data_dict_fp = join(dirname(realpath(__file__)), '..', '..', 'tests',
                        'data', 'table_s2.xlsx')

    def _load_vioscreen(self):
        with open(self.ext_survey_fp, 'rU') as f:
            # -160 is to remove a suffix attached to our survey_ids
            db.store_external_survey(f, 'Vioscreen', separator=',',
                                     survey_id_col='Username', trim='-160')

    def setUp(self):
        # Make sure vioscreen survey exists in DB
        try:
            db.add_external_survey('Vioscreen', 'FFQ', 'http://vioscreen.com')
        except ValueError:
            pass
        self._load_vioscreen()

    def tearDown(self):
        db._clear_table('external_survey_answers', 'ag')
        db._revert_ready(['000023299'])

    def test_clean_and_ambiguous_barcodes(self):
        exp = [(tuple(), dict()),
               (('000000000', '123456789'), dict()),
               (('000000000', '123456789', 'A00000000'),
                {'000000000': ['000000000B', '000000000C'],
                 'A00000000': ['A000000000']})]
        data = [tuple(), ('000000000', '123456789'),
                ('000000000', 'A000000000', '123456789',
                 '000000000B', '000000000C')]
        for dat, (e_clean, e_amb) in zip(data, exp):
            o_clean, o_amb = db._clean_and_ambiguous_barcodes(dat)
            self.assertEqual(o_clean, e_clean)
            self.assertEqual(o_amb, e_amb)

        with self.assertRaises(ValueError):
            db._clean_and_ambiguous_barcodes(['000000', ])

    def test_smooth_survey_pets(self):
        df = pd.DataFrame([], columns=['survey',
                                       'participant_survey_id',
                                       'barcode',
                                       'question',
                                       'answer'])
        exp = pd.DataFrame([], columns=['survey',
                                        'participant_survey_id',
                                        'barcode',
                                        'question',
                                        'answer'])
        obs = db._smooth_survey_pets(df)
        pdt.assert_frame_equal(df, exp)

        df = pd.DataFrame([(2, 'abcd', '123456789', 'foo', 'bar'),
                           (2, 'wxyz', '123456789', 'foo', 'bar'),
                           (2, 'wxyz', '123456789', 'foo2', 'bar2'),
                           (2, 'wxyz', '123459', 'foo3', 'bar3'),
                           (1, 'abcd', '111119', 'foo', 'bar'),
                           (1, 'xxx', '126789', 'foo', 'bar')],
                           columns=['survey',
                                    'participant_survey_id',
                                    'barcode',
                                    'question',
                                    'answer'])
        obs = db._smooth_survey_pets(df)
        self.assertEqual(len(obs[obs.survey == 1]), 2)
        self.assertEqual(len(obs[obs.barcode == '123459']), 16)
        self.assertEqual(len(obs[obs.barcode == '123456789']), 33)
        self.assertEqual(len(obs[obs.participant_survey_id == 'abcd']), 17)
        env_feature = obs[obs.question == 'ENV_FEATURE']
        self.assertEqual(env_feature.answer.value_counts().to_dict(),
                        {'animal-associated habitat': 3})

    def test_smooth_survey_host_invariant(self):
        df = pd.DataFrame([], columns=['survey',
                                       'participant_survey_id',
                                       'barcode',
                                       'question',
                                       'answer'])
        exp = pd.DataFrame([], columns=['survey',
                                        'participant_survey_id',
                                        'barcode',
                                        'question',
                                        'answer'])
        obs = db._smooth_survey_host_invariant(df, is_human=True)
        pdt.assert_frame_equal(df, exp)

        df = pd.DataFrame([(2, 'abcd', '123456789', 'foo', 'bar'),
                           (2, 'wxyz', '123456789', 'foo', 'bar'),
                           (2, 'wxyz', '123456789', 'foo2', 'bar2'),
                           (2, 'wxyz', '123459', 'foo3', 'bar3'),
                           (1, 'abcd', '111119', 'foo', 'bar'),
                           (1, 'xxx', '126789', 'foo', 'bar')],
                           columns=['survey',
                                    'participant_survey_id',
                                    'barcode',
                                    'question',
                                    'answer'])

        # note that smooth_survey_host_invariant returns _only_ the rows to
        # be added
        obs = db._smooth_survey_host_invariant(df[df.survey == 1],
                                               is_human=True)

        self.assertEqual(len(obs[obs.survey == 1]), 30)
        self.assertEqual(len(obs[obs.barcode == '123459']), 0)
        self.assertEqual(len(obs[obs.barcode == '123456789']), 0)
        self.assertEqual(len(obs[obs.participant_survey_id == 'abcd']), 15)
        env_feature = obs[obs.question == 'ENV_FEATURE']
        self.assertEqual(env_feature.answer.value_counts().to_dict(),
                        {'human-associated habitat': 2})

    def test_smooth_survey_human(self):
        df = pd.DataFrame([], columns=['survey',
                                       'participant_survey_id',
                                       'barcode',
                                       'question',
                                       'answer'])
        exp = pd.DataFrame([], columns=['survey',
                                        'participant_survey_id',
                                        'barcode',
                                        'question',
                                        'answer'])
        obs = db._smooth_survey_human(df)
        pdt.assert_frame_equal(df, exp)

        df = pd.DataFrame([(2, 'abcd', '123456789', 'foo', 'bar'),
                           (2, 'wxyz', '123456789', 'foo', 'bar'),
                           (2, 'wxyz', '123456789', 'foo2', 'bar2'),
                           (2, 'wxyz', '123459', 'foo3', 'bar3'),
                           (1, 'abcd', '111119', 'foo', 'bar'),
                           (1, 'xxx', '126789', 'foo', 'bar')],
                           columns=['survey',
                                    'participant_survey_id',
                                    'barcode',
                                    'question',
                                    'answer'])
        obs = db._smooth_survey_human(df)
        self.assertEqual(sorted(obs.columns), sorted(df.columns))

        exp = "This needs to be defined"
        pdt.assert_frame_equal(df, exp)

    def test_smooth_survey_yesno(self):
        df = pd.DataFrame([], columns=['survey',
                                       'participant_survey_id',
                                       'barcode',
                                       'question',
                                       'answer'])
        exp = pd.DataFrame([], columns=['survey',
                                        'participant_survey_id',
                                        'barcode',
                                        'question',
                                        'answer'])
        db._smooth_survey_yesno(df)
        pdt.assert_frame_equal(df, exp)

        df = pd.DataFrame([(1, 'abcd', '123456789', 'foo', 'bar'),
                           (2, 'abcd', '123456789', 'foo', 'bar'),
                           (3, 'abcd', '123456789', 'foo', 'bar'),
                           (4, 'abcd', '123456789', 'foo', 'bar'),
                           (5, 'abcd', '123456789', 'foo', 'bar'),
                           (5, 'abcd', '123456789', 'foo', 'baz'),
                           (5, 'abcd', '123456789', 'foo', 'yes'),
                           (5, 'abcd', '123456789', 'foo', 'Yes'),
                           (5, 'abcd', '123456789', 'foo', 'no'),
                           (5, 'abcd', '123456789', 'foo', 'No'),
                           (1, 'abcd', '123456789', 'foo', 'bar')],
                          columns=['survey',
                                   'participant_survey_id',
                                   'barcode',
                                   'question',
                                   'answer'])

        exp = pd.DataFrame([(1, 'abcd', '123456789', 'foo', 'bar'),
                            (2, 'abcd', '123456789', 'foo', 'bar'),
                            (3, 'abcd', '123456789', 'foo', 'bar'),
                            (4, 'abcd', '123456789', 'foo', 'bar'),
                            (5, 'abcd', '123456789', 'foo', 'bar'),
                            (5, 'abcd', '123456789', 'foo', 'baz'),
                            (5, 'abcd', '123456789', 'foo', 'true'),
                            (5, 'abcd', '123456789', 'foo', 'true'),
                            (5, 'abcd', '123456789', 'foo', 'false'),
                            (5, 'abcd', '123456789', 'foo', 'false'),
                            (1, 'abcd', '123456789', 'foo', 'bar')],
                           columns=['survey',
                                    'participant_survey_id',
                                    'barcode',
                                    'question',
                                    'answer'])
        db._smooth_survey_yesno(df)
        pdt.assert_frame_equal(df, exp)

    def test_smooth_nulls(self):
        df = pd.DataFrame([['unspecified', 'foo'],
                           [u'unspECified', 'bar'],
                           [np.nan, 'baz'],
                           ['stuff', 'qasd']], columns=['answer', 'other'])
        exp = df.copy()
        exp['answer'] = ['Not provided', 'Not provided', 'Not provided',
                         'stuff']
        obs = db._smooth_nulls(df)
        pdt.assert_frame_equal(obs, exp)

    def test_human_create_subset_bmi(self):
        df = pd.DataFrame([[19, 'foo'],
                           [18, 'bar'],
                           [31, 'baz'],
                           [np.nan, 'asd']], columns=['BMI_CORRECTED',
                                                      'other'])
        exp = df.copy()
        exp['SUBSET_BMI'] = [True, False, False, False]
        obs = db._human_create_subset_bmi(df)
        pdt.assert_frame_equal(obs, exp, check_column_type=False)

    def test_human_create_ibd_diagnosis(self):
        df = pd.DataFrame([["Ileal Crohn's Disease", 'foo'],
                           ["Colonic Crohn's Disease", "bar"],
                           ["something", "baz"],
                           ["Ulcerative colitis", "asd"]],
                          columns=['IBD_DIAGNOSIS_REFINED', 'thing'])
        exp = df.copy()
        exp['IBD_DIAGNOSIS'] = ["Crohn's disease", "Crohn's disease",
                                'Not provided', "Ulcerative colitis"]
        obs = db._human_create_ibd_diagnosis(df)
        pdt.assert_frame_equal(obs, exp, check_column_type=False)

    def test_human_create_alcohol_consumption(self):
        # the underlying categorization is pretty loose.
        df = pd.DataFrame([['Never', 'blah'],
                           ['Daily', 'biz']],
                          columns=['ALCOHOL_FREQUENCY', 'other'])
        exp = df.copy()
        exp['ALCOHOL_CONSUMPTION'] = ['No', 'Yes']
        obs = db._human_create_alcohol_consumption(df)
        pdt.assert_frame_equal(obs, exp, check_column_type=False)

    def test_human_create_collection_season(self):
        df = pd.DataFrame([[2, 'bar'],
                           ['thing', 'baz'],
                           [7, 'biz']], columns=['COLLECTION_MONTH', 'other'])
        exp = df.copy()
        exp['COLLECTION_SEASON'] = ['Winter', 'Not provided', 'Summer']
        obs = db._human_create_collection_season(df)
        pdt.assert_frame_equal(obs, exp, check_column_type=False)

    def test_human_normalize_numeric(self):
        df = pd.DataFrame([[10, 20, 30],
                           ['10', '20', '30'],
                           ['foo', 5, 2],
                           [None, np.nan, 'foo']],
                          columns=['HEIGHT_CM', 'WEIGHT_KG', 'other'])
        exp = df.copy()
        exp['HEIGHT_CM'] = [10.0, 10.0, 'Not provided', 'Not provided']
        exp['WEIGHT_KG'] = [20.0, 20.0, 5.0, 'Not provided']
        obs = db._human_normalize_numeric(df)
        pdt.assert_frame_equal(obs, exp)

    def test_human_normalize_height(self):
        df = pd.DataFrame([[10, 'inches'],
                           [20, 'centimeters'],
                           [None, 'inches'],
                           ['', 'inches'],
                           [np.nan, 'inches'],
                           [np.nan, 'centimeters']],
                          columns=['HEIGHT_CM', 'HEIGHT_UNITS'])
        exp = df.copy()
        exp['HEIGHT_CM'] = [10 * 2.54, 20, None, '', np.nan, np.nan]
        exp['HEIGHT_UNITS'] = ['centimeters',
                               'centimeters',
                               'centimeters',
                               'centimeters',
                               'centimeters',
                               'centimeters']

        obs = db._human_normalize_height(df)
        pdt.assert_frame_equal(obs, exp)


    def test_human_normalize_weight(self):
        df = pd.DataFrame([[10, 'pounds'],
                           [20, 'kilograms'],
                           [None, 'pounds'],
                           ['', 'pounds'],
                           [np.nan, 'pounds'],
                           [np.nan, 'kilograms']],
                          columns=['WEIGHT_KG', 'WEIGHT_UNITS'])
        exp = df.copy()
        exp['WEIGHT_KG'] = [10 / 2.20462, 20, None, '', np.nan, np.nan]
        exp['WEIGHT_UNITS'] = ['kilograms',
                               'kilograms',
                               'kilograms',
                               'kilograms',
                               'kilograms',
                               'kilograms']

        obs = db._human_normalize_weight(df)
        pdt.assert_frame_equal(obs, exp)

    def test_human_create_bmi(self):
        df = pd.DataFrame([[50, 10],
                           ['Not provided', 20],
                           [15, 0],
                           [15, 0.0],
                           [0.1, 200]], columns=['HEIGHT_CM', 'WEIGHT_KG'])
        exp = df.copy()
        exp['BMI'] = [10 / (50.0 / 100)**2,
                      'Not provided',
                      'Not provided',
                      'Not provided',
                      200.0 / (0.1 / 100)**2]
        exp['BMI_CORRECTED'] = ['40.00',
                                'Not provided',
                                'Not provided',
                                'Not provided',
                                'Not provided']
        exp['BMI_CAT'] = ['Obese',
                          'Not provided',
                          'Not provided',
                          'Not provided',
                          'Not provided']

        obs = db._human_create_bmi(df)
        pdt.assert_frame_equal(obs, exp, check_column_type=False)

    def test_human_create_age_years(self):
        df = pd.DataFrame([[1, 1970, 2017, 1, 1],
                           ['Not provided', 'Not provided', 2017, 1, 1],
                           [1, 1971, 2017, 1, 1]],
                           columns=['BIRTH_MONTH', 'BIRTH_YEAR',
                                    'COLLECTION_YEAR', 'COLLECTION_MONTH',
                                    'COLLECTION_DAY'])
        exp = df.copy()
        exp['AGE_YEARS'] = ['47', 'Not provided', '46']
        obs = db._human_create_age_years(df)
        pdt.assert_frame_equal(obs, exp, check_column_type=False)

    def test_human_create_age_corrected(self):
        df = pd.DataFrame([(5, 50, 20, 'foo'),
                           (2, 200, 50, 'bar'),
                           (2, 30, 10, 'daily'),
                           (2, 30, 10, 'Never'),
                           (25, 123, 123, 'blah')],
                           columns=['AGE_YEARS',
                                    'HEIGHT_CM',
                                    'WEIGHT_KG',
                                    'ALCOHOL_CONSUMPTION'])
        exp = df.copy()
        exp['AGE_CORRECTED'] = [5, 'Not provided', 'Not provided', 2, 25]
        exp['AGE_CAT'] = ['child', 'Not provided', 'Not provided', 'baby',
                          '20s']
        obs = db._human_create_age_corrected(df)
        pdt.assert_frame_equal(obs, exp, check_column_type=False)

    def test_human_create_sex(self):
        df = pd.DataFrame([['male', 'fooA'],
                           ['Male', 'fooB'],
                           ['female', 'fooC'],
                           ['Female', 'food'],
                           ['mAlE', 'foox'],
                           ['other', 'foox'],  # I dont think this is possible?
                           ['unspecified', 'foox'],
                           [None, 'asd']], columns=['GENDER', 'BLAH'])
        exp = df.copy()
        exp['SEX'] = ['male', 'male', 'female', 'female', 'male',
                      'other', 'unspecified', 'Not provided']
        obs = db._human_create_sex(df)
        pdt.assert_frame_equal(obs, exp, check_column_type=False)

    def test_human_create_subset_age(self):
        exp = pd.DataFrame([(1, 2, 3, 4), (5, 6, 7, 8)], columns=list('abcd'))
        obs = db._human_create_subset_age(exp.copy())
        pdt.assert_frame_equal(obs, exp)
        exp = pd.DataFrame([(1, 2, 3, 4, 15, False),
                            (5, 6, 7, 8, 20, True),
                            (5, 6, 7, 8, 30, True),
                            (5, 6, 7, 8, 90, False)],
                            columns=list('abcd') + ['AGE_YEARS', 'SUBSET_AGE'])
        obs = db._human_create_subset_age(exp.copy())
        pdt.assert_frame_equal(obs, exp)

    def test_human_create_subset_column(self):
        exp = pd.DataFrame([(1, 2, 3, 4), (5, 6, 7, 8)], columns=list('abcd'))
        obs = db._human_create_subset_column(exp.copy(),
                                             'foo', {'thing': 'yes'})
        pdt.assert_frame_equal(obs, exp)

        df = pd.DataFrame([[80, True, True, True, True, True, 'y'],
                           [15, False, True, True, True, True, np.nan],
                           [np.nan] * 7,
                           [np.nan, True, True, True, True, True, False],
                           [np.nan, True, True, True, False, True, False],
                           [np.nan, True, True, np.nan, True, True, False]],
                           columns=['foo', 'SUBSET_AGE',
                                    'SUBSET_DIABETES',
                                    'SUBSET_IBD',
                                    'SUBSET_ANTIBIOTIC_HISTORY',
                                    'SUBSET_BMI', 'bar'])

        exp = pd.DataFrame([[80, True, True, True, True, True, 'y', True],
                            [15, False, True, True, True, True, np.nan, False],
                            [np.nan] * 7 + [False],
                            [np.nan, True, True, True, True, True, False,
                             True],
                            [np.nan, True, True, True, False, True, False,
                             False],
                            [np.nan, True, True, np.nan, True, True, False,
                             False]],
                            columns=['foo', 'SUBSET_AGE',
                                     'SUBSET_DIABETES',
                                     'SUBSET_IBD',
                                     'SUBSET_ANTIBIOTIC_HISTORY',
                                     'SUBSET_BMI', 'bar',
                                     'SUBSET_HEALTHY'])

        obs = db._human_create_subset_column(df.copy(), 'SUBSET_HEALTHY',
                                             {'SUBSET_AGE': True,
                                              'SUBSET_DIABETES': True,
                                              'SUBSET_IBD': True,
                                              'SUBSET_ANTIBIOTIC_HISTORY': True,
                                              'SUBSET_BMI': True})
        pdt.assert_frame_equal(obs, exp, check_column_type=False, check_dtype=False)

    def test_integrate_barcode_information(self):
        df = pd.DataFrame([(1, 'abcd', '123456789', 'foo', 'bar'),
                           (2, 'abcd', '223456789', 'foo', 'bar'),
                           (3, 'xbcd', '223456789', 'foo', 'bar'),
                           (4, 'abcd', '223456789', 'foo', 'bar'),
                           (5, 'abcd', '523456789', 'foo', 'yes'),
                           (5, 'abcd', '523456789', 'foo', 'Yes'),
                           (1, 'abcd', '823456789', 'foo', 'bar')],
                          columns=['survey',
                                   'participant_survey_id',
                                   'barcode',
                                   'question',
                                   'answer'])

        barcode_info = {'123456789': {'sample_date': datetime.datetime(1970, 1, 4),
                                      'sample_time': datetime.time(1, 3),
                                      'site_sampled': 'Stool',
                                      'participant_name': 'bob',
                                      'country': 'USA',
                                      'ZIP_CODE': '12345',
                                      'STATE': 'MA',
                                      'ag_login_id': 'x',
                                      },
                        '223456789': {'sample_date': datetime.datetime(1975, 2, 3),
                                      'sample_time': datetime.time(2, 4),
                                      'site_sampled': 'Forehead',
                                      'participant_name': 'fred',
                                      'country': 'Canada',
                                      'ZIP_CODE': '12345',
                                      'STATE': 'WA',
                                      'ag_login_id': 'y',
                                      },
                        '523456789': {'sample_date': datetime.datetime(1980, 3, 2),
                                      'sample_time': datetime.time(3, 5),
                                      'site_sampled': 'Mouth',
                                      'participant_name': 'derf',
                                      'country': 'USA',
                                      'ZIP_CODE': '12345',
                                      'STATE': 'PA',
                                      'ag_login_id': 'z',
                                      },
                        '823456789': {'sample_date': datetime.datetime(1985, 4, 1),
                                      'sample_time': datetime.time(4, 6),
                                      'site_sampled': 'Stool',
                                      'participant_name': 'derp',
                                      'country': 'USA',
                                      'ZIP_CODE': '12345',
                                      'STATE': 'CA',
                                      'ag_login_id': 'w',
                                      }
                        }

        exp = pd.DataFrame([(1, 'abcd', '123456789', 'foo', 'bar'),
                            (1, 'abcd', '123456789', 'HOST_SUBJECT_ID', sha512('x' + 'bob').hexdigest()),
                            (2, 'abcd', '223456789', 'HOST_SUBJECT_ID', sha512('y' + 'fred').hexdigest()),
                            (3, 'xbcd', '223456789', 'HOST_SUBJECT_ID', sha512('y' + 'fred').hexdigest()),
                            (4, 'abcd', '223456789', 'HOST_SUBJECT_ID', sha512('y' + 'fred').hexdigest()),
                            (5, 'abcd', '523456789', 'HOST_SUBJECT_ID', sha512('z' + 'derf').hexdigest()),
                            (1, 'abcd', '823456789', 'HOST_SUBJECT_ID', sha512('w' + 'derp').hexdigest()),

                            (1, 'abcd', '123456789', 'ZIP_CODE', '12345'),
                            (2, 'abcd', '223456789', 'ZIP_CODE', '12345'),
                            (3, 'xbcd', '223456789', 'ZIP_CODE', '12345'),
                            (4, 'abcd', '223456789', 'ZIP_CODE', '12345'),
                            (5, 'abcd', '523456789', 'ZIP_CODE', '12345'),
                            (1, 'abcd', '823456789', 'ZIP_CODE', '12345'),

                            (1, 'abcd', '123456789', 'STATE', 'MA'),
                            (2, 'abcd', '223456789', 'STATE', 'WA'),
                            (3, 'xbcd', '223456789', 'STATE', 'WA'),
                            (4, 'abcd', '223456789', 'STATE', 'WA'),
                            (5, 'abcd', '523456789', 'STATE', 'PA'),
                            (1, 'abcd', '823456789', 'STATE', 'CA'),

                            (1, 'abcd', '123456789', 'COUNTRY', 'USA'),
                            (2, 'abcd', '223456789', 'COUNTRY', 'USA'),
                            (3, 'xbcd', '223456789', 'COUNTRY', 'USA'),
                            (4, 'abcd', '223456789', 'COUNTRY', 'USA'),
                            (5, 'abcd', '523456789', 'COUNTRY', 'USA'),
                            (1, 'abcd', '823456789', 'COUNTRY', 'USA'),

                            (1, 'abcd', '123456789', 'COUNTRY', "Unspecified"),
                            (2, 'abcd', '223456789', 'COUNTRY', "Unspecified"),
                            (3, 'xbcd', '223456789', 'COUNTRY', "Unspecified"),
                            (4, 'abcd', '223456789', 'COUNTRY', "Unspecified"),
                            (5, 'abcd', '523456789', 'COUNTRY', "Unspecified"),
                            (1, 'abcd', '823456789', 'COUNTRY', "Unspecified"),
                            (1, 'abcd', '123456789', 'STATE', "Unspecified"),
                            (2, 'abcd', '223456789', 'STATE', "Unspecified"),
                            (3, 'xbcd', '223456789', 'STATE', "Unspecified"),
                            (4, 'abcd', '223456789', 'STATE', "Unspecified"),
                            (5, 'abcd', '523456789', 'STATE', "Unspecified"),
                            (1, 'abcd', '823456789', 'STATE', "Unspecified"),
                            (1, 'abcd', '123456789', 'LONGITUDE', "Unspecified"),
                            (2, 'abcd', '223456789', 'LONGITUDE', "Unspecified"),
                            (3, 'xbcd', '223456789', 'LONGITUDE', "Unspecified"),
                            (4, 'abcd', '223456789', 'LONGITUDE', "Unspecified"),
                            (5, 'abcd', '523456789', 'LONGITUDE', "Unspecified"),
                            (1, 'abcd', '823456789', 'LONGITUDE', "Unspecified"),
                            (1, 'abcd', '123456789', 'LATITUDE', "Unspecified"),
                            (2, 'abcd', '223456789', 'LATITUDE', "Unspecified"),
                            (3, 'xbcd', '223456789', 'LATITUDE', "Unspecified"),
                            (4, 'abcd', '223456789', 'LATITUDE', "Unspecified"),
                            (5, 'abcd', '523456789', 'LATITUDE', "Unspecified"),
                            (1, 'abcd', '823456789', 'LATITUDE', "Unspecified"),
                            (1, 'abcd', '123456789', 'ELEVATION', "Unspecified"),
                            (2, 'abcd', '223456789', 'ELEVATION', "Unspecified"),
                            (3, 'xbcd', '223456789', 'ELEVATION', "Unspecified"),
                            (4, 'abcd', '223456789', 'ELEVATION', "Unspecified"),
                            (5, 'abcd', '523456789', 'ELEVATION', "Unspecified"),
                            (1, 'abcd', '823456789', 'ELEVATION', "Unspecified"),
                            (1, 'abcd', '123456789', 'GEO_LOC_NAME', "Unspecified"),
                            (2, 'abcd', '223456789', 'GEO_LOC_NAME', "Unspecified"),
                            (3, 'xbcd', '223456789', 'GEO_LOC_NAME', "Unspecified"),
                            (4, 'abcd', '223456789', 'GEO_LOC_NAME', "Unspecified"),
                            (5, 'abcd', '523456789', 'GEO_LOC_NAME', "Unspecified"),
                            (1, 'abcd', '823456789', 'GEO_LOC_NAME', "Unspecified"),


                            (1, 'abcd', '123456789', 'COLLECTION_YEAR', 1970),
                            (2, 'abcd', '223456789', 'COLLECTION_YEAR', 1975),
                            (3, 'xbcd', '223456789', 'COLLECTION_YEAR', 1975),
                            (4, 'abcd', '223456789', 'COLLECTION_YEAR', 1975),
                            (5, 'abcd', '523456789', 'COLLECTION_YEAR', 1980),
                            (1, 'abcd', '823456789', 'COLLECTION_YEAR', 1985),
                            (1, 'abcd', '123456789', 'COLLECTION_MONTH', 1),
                            (2, 'abcd', '223456789', 'COLLECTION_MONTH', 2),
                            (3, 'xbcd', '223456789', 'COLLECTION_MONTH', 2),
                            (4, 'abcd', '223456789', 'COLLECTION_MONTH', 2),
                            (5, 'abcd', '523456789', 'COLLECTION_MONTH', 3),
                            (1, 'abcd', '823456789', 'COLLECTION_MONTH', 4),
                            (1, 'abcd', '123456789', 'COLLECTION_DAY', 4),
                            (2, 'abcd', '223456789', 'COLLECTION_DAY', 3),
                            (3, 'xbcd', '223456789', 'COLLECTION_DAY', 3),
                            (4, 'abcd', '223456789', 'COLLECTION_DAY', 3),
                            (5, 'abcd', '523456789', 'COLLECTION_DAY', 2),
                            (1, 'abcd', '823456789', 'COLLECTION_DAY', 1),
                            (1, 'abcd', '123456789', 'COLLECTION_HOUR', 1),
                            (2, 'abcd', '223456789', 'COLLECTION_HOUR', 2),
                            (3, 'xbcd', '223456789', 'COLLECTION_HOUR', 2),
                            (4, 'abcd', '223456789', 'COLLECTION_HOUR', 2),
                            (5, 'abcd', '523456789', 'COLLECTION_HOUR', 3),
                            (1, 'abcd', '823456789', 'COLLECTION_HOUR', 4),
                            (1, 'abcd', '123456789', 'COLLECTION_MINUTE', 3),
                            (2, 'abcd', '223456789', 'COLLECTION_MINUTE', 4),
                            (3, 'xbcd', '223456789', 'COLLECTION_MINUTE', 4),
                            (4, 'abcd', '223456789', 'COLLECTION_MINUTE', 4),
                            (5, 'abcd', '523456789', 'COLLECTION_MINUTE', 5),
                            (1, 'abcd', '823456789', 'COLLECTION_MINUTE', 6),
                            (2, 'abcd', '223456789', 'foo', 'bar'),
                            (3, 'xbcd', '223456789', 'foo', 'bar'),
                            (4, 'abcd', '223456789', 'foo', 'bar'),
                            (5, 'abcd', '523456789', 'foo', 'yes'),
                            (5, 'abcd', '523456789', 'foo', 'Yes'),
                            (1, 'abcd', '823456789', 'foo', 'bar'),
                            (1, 'abcd', '123456789', 'BODY_PRODUCT', 'UBERON:feces'),
                            (1, 'abcd', '123456789', 'SAMPLE_TYPE', 'Stool'),
                            (1, 'abcd', '123456789', 'SCIENTIFIC_NAME', 'human gut metagenome'),
                            (1, 'abcd', '123456789', 'TAXON_ID', '408170'),
                            (1, 'abcd', '123456789', 'BODY_HABITAT', 'UBERON:feces'),
                            (1, 'abcd', '123456789', 'ENV_MATERIAL', 'feces'),
                            (1, 'abcd', '123456789', 'ENV_PACKAGE', 'human-gut'),
                            (1, 'abcd', '123456789', 'DESCRIPTION', 'American Gut Project Stool Sample'),
                            (1, 'abcd', '123456789', 'BODY_SITE', 'UBERON:feces'),
                            (2, 'abcd', '223456789', 'BODY_PRODUCT', 'UBERON:sebum'),
                            (2, 'abcd', '223456789', 'SAMPLE_TYPE', 'Forehead'),
                            (2, 'abcd', '223456789', 'SCIENTIFIC_NAME', 'human skin metagenome'),
                            (2, 'abcd', '223456789', 'TAXON_ID', '539655'),
                            (2, 'abcd', '223456789', 'BODY_HABITAT', 'UBERON:skin'),
                            (2, 'abcd', '223456789', 'ENV_MATERIAL', 'sebum'),
                            (2, 'abcd', '223456789', 'ENV_PACKAGE', 'human-skin'),
                            (2, 'abcd', '223456789', 'DESCRIPTION', 'American Gut Project Forehead Sample'),
                            (2, 'abcd', '223456789', 'BODY_SITE', 'UBERON:skin of head'),
                            (3, 'xbcd', '223456789', 'BODY_PRODUCT', 'UBERON:sebum'),
                            (3, 'xbcd', '223456789', 'SAMPLE_TYPE', 'Forehead'),
                            (3, 'xbcd', '223456789', 'SCIENTIFIC_NAME', 'human skin metagenome'),
                            (3, 'xbcd', '223456789', 'TAXON_ID', '539655'),
                            (3, 'xbcd', '223456789', 'BODY_HABITAT', 'UBERON:skin'),
                            (3, 'xbcd', '223456789', 'ENV_MATERIAL', 'sebum'),
                            (3, 'xbcd', '223456789', 'ENV_PACKAGE', 'human-skin'),
                            (3, 'xbcd', '223456789', 'DESCRIPTION', 'American Gut Project Forehead Sample'),
                            (3, 'xbcd', '223456789', 'BODY_SITE', 'UBERON:skin of head'),
                            (4, 'abcd', '223456789', 'BODY_PRODUCT', 'UBERON:sebum'),
                            (4, 'abcd', '223456789', 'SAMPLE_TYPE', 'Forehead'),
                            (4, 'abcd', '223456789', 'SCIENTIFIC_NAME', 'human skin metagenome'),
                            (4, 'abcd', '223456789', 'TAXON_ID', '539655'),
                            (4, 'abcd', '223456789', 'BODY_HABITAT', 'UBERON:skin'),
                            (4, 'abcd', '223456789', 'ENV_MATERIAL', 'sebum'),
                            (4, 'abcd', '223456789', 'ENV_PACKAGE', 'human-skin'),
                            (4, 'abcd', '223456789', 'DESCRIPTION', 'American Gut Project Forehead Sample'),
                            (4, 'abcd', '223456789', 'BODY_SITE', 'UBERON:skin of head'),
                            (5, 'abcd', '523456789', 'BODY_PRODUCT', 'UBERON:saliva'),
                            (5, 'abcd', '523456789', 'SAMPLE_TYPE', 'Mouth'),
                            (5, 'abcd', '523456789', 'SCIENTIFIC_NAME', 'human oral metagenome'),
                            (5, 'abcd', '523456789', 'TAXON_ID', '447426'),
                            (5, 'abcd', '523456789', 'BODY_HABITAT', 'UBERON:oral cavity'),
                            (5, 'abcd', '523456789', 'ENV_MATERIAL', 'saliva'),
                            (5, 'abcd', '523456789', 'ENV_PACKAGE', 'human-oral'),
                            (5, 'abcd', '523456789', 'DESCRIPTION', 'American Gut Project Mouth Sample'),
                            (5, 'abcd', '523456789', 'BODY_SITE', 'UBERON:tongue'),
                            (1, 'abcd', '823456789', 'BODY_PRODUCT', 'UBERON:feces'),
                            (1, 'abcd', '823456789', 'SAMPLE_TYPE', 'Stool'),
                            (1, 'abcd', '823456789', 'SCIENTIFIC_NAME', 'human gut metagenome'),
                            (1, 'abcd', '823456789', 'TAXON_ID', '408170'),
                            (1, 'abcd', '823456789', 'BODY_HABITAT', 'UBERON:feces'),
                            (1, 'abcd', '823456789', 'ENV_MATERIAL', 'feces'),
                            (1, 'abcd', '823456789', 'ENV_PACKAGE', 'human-gut'),
                            (1, 'abcd', '823456789', 'DESCRIPTION', 'American Gut Project Stool Sample'),
                            (1, 'abcd', '823456789', 'BODY_SITE', 'UBERON:feces')],
                           columns=['survey',
                                    'participant_survey_id',
                                    'barcode',
                                    'question',
                                    'answer'])
        order = ['survey', 'participant_survey_id', 'barcode', 'question',]
        obs = db._integrate_barcode_information(df, barcode_info)

        obs = obs.sort_values(order)
        exp = exp.sort_values(order)

        # couldn't manage to ignore the index which does not contain useful
        # information, so explicitly resetting
        obs.index = range(len(obs))
        exp.index = range(len(exp))

        # NOTE: this test will fail until the geocoder works.
        pdt.assert_frame_equal(obs, exp)

    def test_human_create_economic_census_regions(self):
        df = pd.DataFrame([[80, 'CA'],
                           [15, np.nan],
                           [np.nan, 'UK']],
                           columns=['foo', 'STATE'])
        exp = pd.DataFrame([[80, 'CA', 'West', 'Far West'],
                            [15, np.nan, 'Not provided', 'Not provided'],
                            [np.nan, 'UK', 'Not provided', 'Not provided']],
                            columns=['foo', 'STATE', 'CENSUS_REGION',
                                     'ECONOMIC_REGION'])
        obs = db._human_create_economic_census_regions(df)
        pdt.assert_frame_equal(obs, exp, check_column_type=False)

    def test_get_vioscreen_survey_answers(self):
        exp = pd.DataFrame([], columns=['survey',
                                        'participant_survey_id',
                                        'barcode',
                                        'question',
                                        'answer'])
        obs = db._get_vioscreen_survey_answers(['doesnotexist', ])
        pdt.assert_frame_equal(obs, exp, check_index_type=False, check_dtype=False)

        obs = db._get_vioscreen_survey_answers(['000023299',   # dup sid
                                                '000023300',   # dup sid
                                                '000004216',   # no responses
                                                '000018046'])  # has resp

        self.assertEqual(len(obs['participant_survey_id'].unique()), 2)
        self.assertEqual(set(obs['barcode'].unique()), {'000023299',
                                                        '000023300',
                                                        '000018046'})
        obs_q = set(obs['question'].unique())
        spot = obs_q.intersection({'VIOSCREEN_VITA_IU',
                                   'VIOSCREEN_GLYCINE',
                                   'VIOSCREEN_V_ORANGE'})
        self.assertEqual(len(obs_q.intersection(spot)), 3)
        self.assertEqual(len(obs_q) * 3, len(obs))

        # apparently no these surveys have no vitd2 in the raw data
        obs_vitd2 = obs[obs['question'] == 'VIOSCREEN_VITD2'].answer.tolist()
        self.assertEqual(obs_vitd2, ['0', '0', '0'])

        obs_zinc = obs[obs['question'] == 'VIOSCREEN_ZINC'].answer.tolist()
        self.assertEqual(obs_zinc, ['15.7077523', '15.7077523', '8.278876819'])

    def test_get_single_survey_answers(self):
        exp = pd.DataFrame([], columns=['survey',
                                        'participant_survey_id',
                                        'barcode',
                                        'question',
                                        'answer'])
        obs = db._get_single_survey_answers(['doesnotexist', ])
        pdt.assert_frame_equal(obs, exp)

        obs = db._get_single_survey_answers(['000004216'])

        # spot check
        handed = obs[obs['question'] == 'DOMINANT_HAND']
        self.assertEqual(handed['answer'].values[0],
                         'I am right handed')

        diet = obs[obs['question'] == 'DIET_TYPE']
        self.assertEqual(diet['answer'].values[0],
                         'Omnivore')

        # all questions are unique
        self.assertEqual(len(obs.question.unique()),
                         len(obs))

    def test_get_multiple_survey_answers(self):
        exp = pd.DataFrame([], columns=['survey',
                                        'participant_survey_id',
                                        'barcode',
                                        'question',
                                        'answer'])
        obs = db._get_multiple_survey_answers(['doesnotexist', ])
        pdt.assert_frame_equal(obs, exp)

        obs = db._get_multiple_survey_answers(['000004216'])

        # spot check
        # ALCOHOL_TYPES gets expanded, and the question itself
        # should not be present in the output
        self.assertNotIn('ALCOHOL_TYPES', obs.question)

        # but the responses to it should be
        alc_shortnames = {'ALCOHOL_TYPES_BEERCIDER': 'false',
                          'ALCOHOL_TYPES_SOUR_BEERS': 'false',
                          'ALCOHOL_TYPES_WHITE_WINE': 'false',
                          'ALCOHOL_TYPES_RED_WINE': 'false',
                          'ALCOHOL_TYPES_SPIRITSHARD_ALCOHOL': 'false',
                          'ALCOHOL_TYPES_UNSPECIFIED': 'true'}
	alc = obs[obs['question'].isin(alc_shortnames.keys())]
        self.assertEqual(alc.shape, (6, 5))
        for q, r in alc_shortnames.items():
            o = alc[alc['question'] == q].answer.values[0]
            self.assertEqual(o, r)

        # all questions are unique
        self.assertEqual(len(obs.question.unique()),
                         len(obs))

    def test_get_other_survey_answers(self):
        exp = pd.DataFrame([], columns=['survey',
                                        'participant_survey_id',
                                        'barcode',
                                        'question',
                                        'answer'])
        obs = db._get_other_survey_answers(['doesnotexist', ])
        pdt.assert_frame_equal(obs, exp)

        obs = db._get_other_survey_answers(['000004216'])
        weight = obs[obs.question == 'WEIGHT_KG']
        self.assertTrue(weight.answer.values[0].startswith('Free text'))

        # all questions are unique
        self.assertEqual(len(obs.question.unique()),
                         len(obs))

    def test_get_surveys(self):
        exp = pd.DataFrame([], columns=['survey',
                                        'participant_survey_id',
                                        'barcode',
                                        'question',
                                        'answer'],
                           index=pd.RangeIndex(0, 0, 1))
        obs = db.get_surveys(['doesnotexist', ])
        pdt.assert_frame_equal(obs, exp)

        base = db.get_surveys(['000004216'])
        reps = db.get_surveys(['000004216A', '000004216B'])

        base_a = base.copy()
        base_a['barcode'] = '000004216A'
        base_b = base.copy()
        base_b['barcode'] = '000004216B'

        exp = pd.concat([base_a, base_b], ignore_index=True)
        exp = exp.sort_values(['barcode', 'question'])
        obs = reps.sort_values(['barcode', 'question'])

        pdt.assert_frame_equal(obs.reset_index(drop=True),
                               exp.reset_index(drop=True))

    def test_sync_with_data_dictionary(self):
        d = {'some_category': u'some response',
             'another_category': 123.0,
             'chickenpox': u'yes',
             'consume_animal_products_abx': u'No',
             'csection': u'not sure',
             'dog': u'Yes',
             'cat': True,
             'lactose': False,
             'other_supplement_frequency': 'Unspecified',
             'lowgrain_diet_type': u'no'}
        exp = {'some_category': u'some response',
               'another_category': 123.0,
               'chickenpox': u'true',
               'consume_animal_products_abx': u'false',
               'csection': u'Not sure',
               'dog': u'true',
               'cat': u'true',
               'lactose': u'false',
               'other_supplement_frequency': u'Not provided',
               'lowgrain_diet_type': u'false'}
        db._sync_with_data_dictionary(d, 'Not provided', 'Not applicable')
        self.assertEqual(d, exp)

    def test_pulldown_data_dictionary_check(self): # noqa: max-complexity(20)
        dd = pd.ExcelFile(self.data_dict_fp)
        primary = dd.parse("Primary Survey")
        vioscreen = dd.parse('Vioscreen FFQ')

        obs, f = db.pulldown(['000029429', '000018046', '000023299',
                              '000023300', '000010863', '000021994',
                              '000010863', '000023772', '000023714',
                              '000001166', '000014401', '000014889',
                              '000041833', '000021693',
                              '000023576', '000013287',
                              '000001586', '000038207'])
        md = pd.read_csv(
            StringIO(obs[1]), delimiter='\t', dtype=str, encoding='utf-8')
        md.columns = [c.lower() for c in md.columns]

        # in data dictionary but not in metadata pulldown
        missing_headers = []
        for c in primary['Column header']:
            if c not in md.columns:
                missing_headers.append(c)
        for c in vioscreen['Column header']:
            if c not in md.columns:
                missing_headers.append(c)

        if missing_headers:
            self.fail("The following headers are missing: %s"
                      % ','.join(missing_headers))

        # in pulldown but not in data dictionary
        missing_headers = []
        no_data = []

        # the specific responses (e.g., alcohol_types_beercider) are stored
        # instead
        ignore = {'alcohol_types', 'allergic_to', 'non_food_allergies',
                  'specialized_diet', 'mental_illness_type'}
        for c in md.columns:
            observed_set = set(md[c].unique())
            if observed_set == {'Missing: Not provided', 'Not applicable'}:
                no_data.append(c)
            elif c.startswith('vioscreen'):
                if c not in vioscreen['Column header'].values \
                        and c not in ignore:
                    missing_headers.append(c)
            else:
                if c not in primary['Column header'].values \
                        and c not in ignore:
                    missing_headers.append(c)
        if missing_headers:
            self.fail("The following headers are unknown: %s"
                      % ','.join(missing_headers))

        boolean_issues = []
        unexp_values = []
        for idx, row in primary.iterrows():
            if isinstance(row['Expected values'], (str, unicode)) \
                    and '|' in row['Expected values']:
                response_set = {s.strip().strip('"').strip("'")
                                for s in row['Expected values'].split('|')}
                response_set.add("Not provided")
                response_set.add('Not applicable')
                bv = row['Blank value']
                if isinstance(bv, str):
                    response_set.add(bv.strip().strip("'").strip('"'))

                observed_set = set(md[row['Column header']].unique())
                if not observed_set.issubset(response_set):
                    if (observed_set - response_set) == {'Yes', 'No'}:
                        boolean_issues.append(row['Column header'])
                    else:
                        unexp_values.append(row['Column header'])

        if boolean_issues:
            self.fail("The following headers had boolean issues: %s"
                      % ','.join(boolean_issues))
        if unexp_values:
            self.fail("The following headers had unexpected values: %s"
                      % ','.join(unexp_values))

    def test_pulldown_third_party(self):
        barcodes = ['000029429', '000018046', '000023299', '000023300']
        # Test without third party
        obs, failures = db.pulldown(barcodes)

        # Parse the metadata into a pandas dataframe to test some invariants
        # This tests does not ensure that the columns have the exact value
        # but at least ensure that the contents looks as expected
        survey_df = pd.read_csv(
            StringIO(obs[1]), delimiter='\t', dtype=str, encoding='utf-8')
        survey_df.set_index('sample_name', inplace=True, drop=True)

        # Make sure that the prohibited columns from EBI are not in the
        # pulldown
        self.assertEqual(set(survey_df.columns).intersection(ebi_remove),
                         set())

        freq_accepted_vals = {
            'Never', 'Rarely (a few times/month)',
            'Regularly (3-5 times/week)', 'Occasionally (1-2 times/week)',
            'Not provided', 'Daily'}

        freq_cols = ['ALCOHOL_FREQUENCY', 'PROBIOTIC_FREQUENCY',
                     'ONE_LITER_OF_WATER_A_DAY_FREQUENCY', 'POOL_FREQUENCY',
                     'FLOSSING_FREQUENCY', 'COSMETICS_FREQUENCY']

        for col in freq_cols:
            vals = set(survey_df[col])
            self.assertTrue(all([x in freq_accepted_vals for x in vals]))

        # This astype is making sure that the values in the BMI column are
        # values that can be casted to float.
        survey_df[survey_df.BMI != 'Not provided'] .BMI.astype(float)

        body_product_values = set(survey_df.BODY_PRODUCT)
        self.assertTrue(all([x.startswith('UBERON') or x == 'Not provided'
                             for x in body_product_values]))

        survey = obs[1]
        self.assertFalse('VIOSCREEN' in survey)

        obs, _ = db.pulldown(barcodes, blanks=['BLANK.01'])
        survey = obs[1]
        self.assertFalse('VIOSCREEN' in survey)
        self.assertTrue('BLANK.01' in survey)

        # Test with third party
        obs, _ = db.pulldown(barcodes, external=['Vioscreen'])
        survey = obs[1]
        self.assertTrue('VIOSCREEN' in survey)

        obs, _ = db.pulldown(barcodes, blanks=['BLANK.01'],
                             external=['Vioscreen'])
        survey = obs[1]
        self.assertTrue('VIOSCREEN' in survey)
        self.assertTrue('BLANK.01' in survey)

    def test_check_consent(self):
        consent, fail = db.check_consent(['000027561', '000001124', '0000000'])
        self.assertEqual(consent, ['000027561'])
        self.assertEqual(fail, {'0000000': 'Not an AG barcode',
                                '000001124': 'Sample not logged'})

    def test_get_unconsented(self):
        obs = db.get_unconsented()
        # we don't know the actual number independent of DB version, but we can
        # assume that we have a certain amount of those barcodes.
        self.assertTrue(len(obs) >= 100)

        # we cannot know which barcodes are unconsented without executing the
        # db function itself. Thus, for unit tests, we should only check data
        # types.
        self.assertTrue(obs[0][0].isdigit())
        self.assertTrue(isinstance(obs[0][1], datetime.date))
        self.assertTrue(isinstance(obs[0][2], str))

    def test_search_kits(self):
        # obtain current test data from DB
        ag_login_id = 'd8592c74-7cf9-2135-e040-8a80115d6401'
        kits = db.get_kit_info_by_login(ag_login_id)

        # check if ag_login_id is regain with supplied_kit_id
        obs = db.search_kits(kits[0]['supplied_kit_id'])
        self.assertEqual([ag_login_id], obs)

        # check if kit_id is found by search
        obs = db.search_kits('e1934dfe-8537-6dce-e040-8a80115d2da9')
        self.assertEqual(['e1934ceb-6e92-c36a-e040-8a80115d2d64'], obs)

        # check that a non existing kit is not found
        obs = db.search_kits('990001124')
        self.assertEqual([], obs)

    def test_get_barcodes_with_results(self):
        obs = db.get_barcodes_with_results()
        exp = ['000023299']
        self.assertEqual(obs, exp)

    def test_mark_results_ready(self):
        db._revert_ready(['000023299'])
        obs = db.get_ag_barcode_details(['000001072', '000023299'])
        self.assertEqual(obs['000023299']['results_ready'], None)
        self.assertEqual(obs['000001072']['results_ready'], 'Y')

        obs = db.mark_results_ready(['000001072', '000023299'], debug=True)
        self.assertEqual(obs['new_bcs'], ('000023299', ))
        self.assertEqual(obs['mail']['mimetext']['To'],
                         'americangut@gmail.com')
        self.assertEqual(obs['mail']['mimetext']['From'], '')
        self.assertEqual(obs['mail']['mimetext']['Subject'],
                         'Your American/British Gut results are ready')
        # don't compare name, since it is scrubbed to random chars
        self.assertEqual(obs['mail']['recipients'][0],
                         'americangut@gmail.com')

        obs = db.get_ag_barcode_details(['000001072', '000023299'])
        self.assertEqual(obs['000023299']['results_ready'], 'Y')
        self.assertEqual(obs['000001072']['results_ready'], 'Y')

    def test_get_access_levels_user(self):
        # insert a fresh new user into DB.
        email = 'testmail@testdomain.com'
        password = ('$2a$10$2.6Y9HmBqUFmSvKCjWmBte70'
                    'WF.zd3h4VqbhLMQK1xP67Aj3rei86')
        sql = """INSERT INTO ag.labadmin_users (email, password)
                 VALUES (%s, %s)"""
        db._con.execute(sql, [email, password])

        obs = db.get_access_levels_user(email)
        self.assertItemsEqual(obs, [])

        db.alter_access_levels(email, [1, 6])
        obs = db.get_access_levels_user(email)
        self.assertItemsEqual(obs, [[1, 'Barcodes'], [6, 'Search']])

        db.alter_access_levels(email, [])
        obs = db.get_access_levels_user(email)
        self.assertItemsEqual(obs, [])

        # Remove test user from DB.
        sql = """DELETE FROM ag.labadmin_users WHERE email=%s"""
        db._con.execute(sql, [email])

    def test_get_users(self):
        obs = db.get_users()
        exp = 'test'
        self.assertIn(exp, obs)

    def test_get_access_levels(self):
        obs = db.get_access_levels()
        exp = [[1, 'Barcodes'], [2, 'AG kits'], [3, 'Scan Barcodes'],
               [4, 'External surveys'], [5, 'Metadata Pulldown'],
               [6, 'Search'], [7, 'Admin']]
        self.assertEqual(obs, exp)

    def test_participant_names(self):
        obs = db.participant_names()
        self.assertTrue(len(obs) >= 8237)
        self.assertIn('000027561', map(lambda x: x[0], obs))

    def test_search_barcodes(self):
        obs = db.search_barcodes('000001124')
        self.assertEqual(obs, ['d8592c74-7c27-2135-e040-8a80115d6401'])

        ag_login_id = "d8592c74-9491-2135-e040-8a80115d6401"
        names = db.ut_get_participant_names_from_ag_login_id(ag_login_id)

        obs = []
        for name in names:
            obs.extend(db.search_barcodes(name))
        self.assertTrue(ag_login_id in obs)

    def test_getAGBarcodeDetails(self):
        obs = db.getAGBarcodeDetails('000018046')
        exp = {'status': 'Received',
               'ag_kit_id': '0060a301-e5c0-6a4e-e050-8a800c5d49b7',
               'barcode': '000018046',
               'environment_sampled': None,
               # 'name': 'REMOVED',
               'ag_kit_barcode_id': '0060a301-e5c1-6a4e-e050-8a800c5d49b7',
               'sample_time': datetime.time(11, 15),
               # 'notes': 'REMOVED',
               'overloaded': 'N',
               'withdrawn': None,  # 'email': 'REMOVED',
               'other': 'N',
               # 'deposited': False,
               # 'participant_name': 'REMOVED-0',
               'refunded': None, 'moldy': 'N',
               'sample_date': datetime.date(2014, 8, 13),
               'date_of_last_email': datetime.date(2014, 8, 15),
               # 'other_text': 'REMOVED',
               'site_sampled': 'Stool'}
        # only look at those fields, that are not subject to scrubbing
        self.assertEqual({k: obs[k] for k in exp}, exp)

    def test_get_barcode_info_by_kit_id(self):
        obs = db.get_barcode_info_by_kit_id(
            '0060a301-e5c0-6a4e-e050-8a800c5d49b7')[0]
        exp = {'ag_kit_id': '0060a301-e5c0-6a4e-e050-8a800c5d49b7',
               'environment_sampled': None,
               'sample_time': datetime.time(11, 15),
               # 'notes': 'REMOVED',
               'barcode': '000018046',
               'results_ready': 'Y',
               'refunded': None,
               # 'participant_name': 'REMOVED-0',
               'ag_kit_barcode_id': '0060a301-e5c1-6a4e-e050-8a800c5d49b7',
               'sample_date': datetime.date(2014, 8, 13),
               'withdrawn': None,
               'site_sampled': 'Stool'}
        # only look at those fields, that are not subject to scrubbing
        self.assertEqual({k: obs[k] for k in exp}, exp)

    def test_getHumanParticipants(self):
        i = "d8592c74-9694-2135-e040-8a80115d6401"
        res = db.getHumanParticipants(i)
        # we can't compare to scrubbed participant names, thus we only check
        # number of names.
        self.assertTrue(len(res) >= 4)

    def test_getHumanParticipantsNotPresent(self):
        i = '00000000-0000-0000-0000-000000000000'
        res = db.getHumanParticipants(i)
        self.assertEqual(res, [])

    def test_getAnimalParticipants(self):
        i = "ed5ab96f-fe3b-ead5-e040-8a80115d1c4b"
        res = db.getAnimalParticipants(i)
        # we can't compare to scrubbed participant names, thus we only check
        # number of names.
        self.assertTrue(len(res) == 1)

    def test_getAnimalParticipantsNotPresent(self):
        i = "00711b0a-67d6-0fed-e050-8a800c5d7570"
        res = db.getAnimalParticipants(i)
        self.assertEqual(res, [])

    def test_get_ag_barcode_details(self):
        obs = db.get_ag_barcode_details(['000018046'])
        ag_login_id = '0060a301-e5bf-6a4e-e050-8a800c5d49b7'
        exp = {'000018046': {
               'ag_kit_barcode_id': '0060a301-e5c1-6a4e-e050-8a800c5d49b7',
               'verification_email_sent': 'n',
               'pass_reset_code': None,
               'vioscreen_status': 3,
               'sample_barcode_file': '000018046.jpg',
               'environment_sampled': None,
               'supplied_kit_id': db.ut_get_supplied_kit_id(ag_login_id),
               'withdrawn': None,
               'kit_verified': 'y',
               # 'city': 'REMOVED',
               'ag_kit_id': '0060a301-e5c0-6a4e-e050-8a800c5d49b7',
               # 'zip': 'REMOVED',
               'ag_login_id': ag_login_id,
               # 'state': 'REMOVED',
               'results_ready': 'Y',
               'moldy': 'N',
               # The key 'registered_on' is a time stamp when the database is
               # created. It is unique per deployment.
               # 'registered_on': datetime.datetime(2016, 8, 17, 10, 47, 2,
               #                                   713292),
               # 'kit_password': ('$2a$10$2.6Y9HmBqUFmSvKCjWmBte70WF.zd3h4Vqb'
               #                  'hLMQK1xP67Aj3rei86'),
               # 'deposited': False,
               'sample_date': datetime.date(2014, 8, 13),
               # 'email': 'REMOVED',
               'print_results': False,
               'open_humans_token': None,
               # 'elevation': 0.0,
               'refunded': None,
               # 'other_text': 'REMOVED',
               'barcode': '000018046',
               'swabs_per_kit': 1L,
               # 'kit_verification_code': '60260',
               # 'latitude': 0.0,
               'cannot_geocode': None,
               # 'address': 'REMOVED',
               'date_of_last_email': datetime.date(2014, 8, 15),
               'site_sampled': 'Stool',
               # 'name': 'REMOVED',
               'sample_time': datetime.time(11, 15),
               # 'notes': 'REMOVED',
               'overloaded': 'N',
               # 'longitude': 0.0,
               'pass_reset_time': None,
               # 'country': 'REMOVED',
               'survey_id': '084532330aca5885',
               'other': 'N',
               'sample_barcode_file_md5': None}}
        participant_names = db.ut_get_participant_names_from_ag_login_id(
            ag_login_id)
        for key in obs:
            del(obs[key]['registered_on'])
            # only look at those fields, that are not subject to scrubbing
            self.assertEqual({k: obs[key][k] for k in exp[key]}, exp[key])
            self.assertIn(obs[key]['participant_name'], participant_names)

    def test_list_ag_surveys(self):
        truth = [(-1, 'Personal Information', True),
                 (-2, 'Pet Information', True),
                 (-3, 'Fermented Foods', True),
                 (-4, 'Surfers', True),
                 (-5, 'Personal_Microbiome', True)]
        self.assertItemsEqual(db.list_ag_surveys(), truth)

        truth = [(-1, 'Personal Information', False),
                 (-2, 'Pet Information', True),
                 (-3, 'Fermented Foods', False),
                 (-4, 'Surfers', True),
                 (-5, 'Personal_Microbiome', False)]
        self.assertItemsEqual(db.list_ag_surveys([-2, -4]), truth)

    def test_scrubb_pet_freetext(self):
        # we had the problem that survey question 150 = 'pets_other_freetext'
        # was exported for pulldown, but it has the potential to carry personal
        # information.

        # this is a barcode where an answer to this question is stored in DB
        barcodes = ['000037487']

        # get free text value from DB
        all_survey_info = db.get_surveys(barcodes)
        freetextvalue = all_survey_info[1]['000037487']['pets_other_freetext']

        # make sure free text value does NOT show up in pulldown
        obs_pulldown = db.pulldown(barcodes)[0]
        for row in obs_pulldown.keys():
            self.assertNotIn(freetextvalue, obs_pulldown[row])


if __name__ == "__main__":
    main()
