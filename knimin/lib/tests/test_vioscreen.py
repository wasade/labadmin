from unittest import TestCase, main
from os.path import join, dirname, realpath
from six import StringIO

from knimin.lib.vioscreen import VioscreenHandler


class TestVioscreenHandler(TestCase):
    def setUp(self):
        self.vio = VioscreenHandler()
    
    def tearDown(self):
        del self.vio

    def test_get_token(self):
        token = self.vio.get_token()
        self.assertEqual(len(token), 259)

    def test_get_users(self):
        users = self.vio.get_users()
        self.assertIsNotNone(users)
        self.assertIsNotNone(users['users'])

        user = users['users'][0]
        res = user.keys()
        exp = [u'username',
               u'weight',
               u'firstname',
               u'displayUnits',
               u'middlename',
               u'lastname',
               u'activityLevel',
               u'created',
               u'subjectId',
               u'email',
               u'height',
               u'dateOfBirth',
               u'gender',
               u'timeZone',
               u'guid',
               u'id']
        for i in res:
            self.assertIn(i, exp)

    def test_pull_vioscreen_data_inval_barcode(self):
        with self.assertRaises(ValueError):
            self.vio.pull_vioscreen_data('notbarcode')

    # I don't know what surveys will be in the test database
    #def test_get_init_surveys(self):

    #def test_update_status(self):

    #def test_insert_survey(self):

    #def test_get_vio_survey_ids_not_in_ag(self):

    def test_tidyfy(self):
        username = 'testuser'
        data = [{'amount': 10,
                 'code': u'substance_a',
                 'description': u'Substance A',
                 'units': u'mg',
                 'valueType': u'Amount'},
                {'amount': 20,
                 'code': u'substance_b',
                 'description': u'Substance B',
                 'units': u'g',
                 'valueType': u'Amount'},
                {'amount': 30,
                 'code': u'substance_c',
                 'description': u'Substance C'}]
        tidy_data = self.vio.tidyfy(username, data)
        for row in tidy_data:
            self.assertIn(username, row.keys())
            del row['survey_id']
        self.assertEqual(data, data2)

    def test_get_session_data(self):
        session_id = u'000ada854d4f45f5abda90ccade7f0a8'
        payload = 'foodcomponents'
    

    #def test_sync_vioscreen(self):

    #def test_pull_vioscreen_data(self):



if __name__ == "__main__":
    main()
