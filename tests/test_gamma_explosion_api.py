"""
tests/test_gamma_explosion_api.py
=============================================================================
Integration tests for /api/gamma/explosion endpoint in DataServer.py.
=============================================================================
"""

import unittest
import json
from DataServer import app

class TestGammaExplosionAPI(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_api_gamma_explosion_endpoint(self):
        resp = self.client.get('/api/gamma/explosion')
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data.decode('utf-8'))
        self.assertTrue(data.get('ok'))
        self.assertIn('active_pins', data)
        self.assertIn('retest_absorptions', data)
        self.assertIn('explosion_targets', data)
        self.assertIsInstance(data['active_pins'], list)
        self.assertIsInstance(data['retest_absorptions'], list)
        self.assertIsInstance(data['explosion_targets'], dict)

if __name__ == '__main__':
    unittest.main()
