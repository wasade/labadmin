# -----------------------------------------------------------------------------
# Copyright (c) 2014--, The LabAdmin Development Team.
#
# Distributed under the terms of the BSD 3-clause License.
#
# The full license is in the file LICENSE, distributed with this software.
# -----------------------------------------------------------------------------
from unittest import main
from functools import partial

from tornado.escape import url_unescape, json_decode

from knimin.tests.tornado_test_base import TestHandlerBase
from knimin import db


class TestPMCreatePlateHandler(TestHandlerBase):
    def test_get_not_authed(self):
        response = self.get('/pm_create_plate/')
        self.assertEqual(response.code, 200)
        self.assertTrue(
            response.effective_url.endswith('?next=%2Fpm_create_plate%2F'))

    def test_get(self):
        self.mock_login_admin()
        response = self.get('/pm_create_plate/')
        self.assertEqual(response.code, 200)
        # Checl that the page is not empty
        self.assertIn('<label><h3>Create new plate</h3></label>',
                      response.body)

    def test_post(self):
        self.mock_login_admin()
        db.create_study(9999, 'LabAdmin test project', 'LTP', 'KL9999')
        self._clean_up_funcs.append(partial(db.delete_study, 9999))
        data = {'plate_type': db.get_plate_types()[0]['id'],
                'studies': [9999],
                'plate_name': 'Test plate 1'}
        response = self.post('/pm_create_plate/', data=data)

        # The new plate id is encoded in the url, as the last value
        # after the last '/' character
        plate_id = url_unescape(response.effective_url).rsplit('/', 2)[1]

        # Using insert here to make sure that this clean up operation
        # is executed before the study one is done
        self._clean_up_funcs.insert(
            0, partial(db.delete_sample_plate, plate_id))

        self.assertEqual(response.code, 200)

        obs = db.read_sample_plate(plate_id)
        # Remove the data as it is not deterministic and its correctness is
        # tested elsewhere
        del obs['created_on']
        exp = {'name': 'Test plate 1', 'plate_type_id': data['plate_type'],
               'email': 'test', 'notes': None, 'studies': [9999]}
        self.assertEqual(obs, exp)


class TestPMPlateNameCheckerHandler(TestHandlerBase):
    def test_get_not_authed(self):
        response = self.get('/pm_sample_plate/name_check?name=TestPlate')
        self.assertEqual(response.code, 200)
        self.assertTrue(
            response.effective_url.endswith(
                '?next=%2Fpm_sample_plate%2Fname_check%3Fname%3DTestPlate'))

    def test_get(self):
        self.mock_login_admin()
        response = self.get('/pm_sample_plate/name_check?name=TestPlate')
        self.assertEqual(response.code, 404)
        # Checl that the page is not empty
        self.assertEqual(json_decode(response.body), {'result': False})

        db.create_study(9999, 'LabAdmin test project', 'LTP', 'KL9999')
        self._clean_up_funcs.append(partial(db.delete_study, 9999))
        plate_id = db.create_sample_plate(
            'TestPlate', db.get_plate_types()[0]['id'], 'test', [9999])
        self._clean_up_funcs.insert(
            0, partial(db.delete_sample_plate, plate_id))
        response = self.get('/pm_sample_plate/name_check?name=TestPlate')
        self.assertEqual(response.code, 200)
        # Checl that the page is not empty
        self.assertEqual(json_decode(response.body), {'result': True})


if __name__ == '__main__':
    main()
