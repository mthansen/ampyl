#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created August 2026.

@author: M.T. Hansen
"""

###############################################################################
#
# test_readme.py
#
# MIT License
# Copyright (c) 2022 Maxwell T. Hansen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
###############################################################################

import contextlib
import io
import os
import re
import unittest

slow_test = unittest.skipUnless(
    os.environ.get('AMPYL_SLOW_TESTS') == '1',
    'slow test: set AMPYL_SLOW_TESTS=1 to run')

README_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'README.md')


@slow_test
class TestReadmeExample(unittest.TestCase):
    """Class to test the README example."""

    def get_example_code(self):
        """Extract the python code block under the Example heading."""
        with open(README_PATH, 'r', encoding='utf-8') as f:
            readme = f.read()
        example_section = readme.split('## Example', 1)[1]
        match = re.search(r'```python\n(.*?)```', example_section, re.DOTALL)
        self.assertIsNotNone(match, 'no python code block under ## Example')
        return match.group(1)

    def test_readme_example(self):
        """Run the README example and check the printed root."""
        code = self.get_example_code()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(code, {'__name__': '__readme_example__'})
        printed = stdout.getvalue().strip()
        root = float(printed.split()[-1])
        self.assertAlmostEqual(root, 3.031816, places=5)


class Template(unittest.TestCase):
    """Test."""

    def setUp(self):
        """Exectue set-up."""
        pass

    def tearDown(self):
        """Execute tear-down."""
        pass

    def __example(self, x):
        return x

    def test(self):
        """Example test."""
        self.assertEqual(10.0, self.__example(10.0))


if __name__ == '__main__':
    unittest.main()
