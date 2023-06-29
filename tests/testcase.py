import unittest
import numpy as np

P = 10 # Precision

def equality_error_message(given, expected):
    return '\nReturned: {0}.\nExpected: {1}'.format(given, expected)

class TestCase(unittest.TestCase):
    def assertRoundEqual(self, a, b):
        if type(a) == float:
            a = round(a, P)
        else:
            a = np.round(a, P)

        if type(b) == float:
            b = round(b, P)
        else:
            b = np.round(b, P)
        
        self.assertEqual(
            a,
            b,
            msg=equality_error_message(a, b)
        )
    
    def assertNumpyEqual(self, given, expected):
        self.assertTrue(
            np.all(given.round(P) == expected.round(P)),
            msg=equality_error_message(given, expected)
        )