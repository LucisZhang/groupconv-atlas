"""Keep measurement identities independent of Finder and Python cache activity."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from bench.run import source_identity

class SourceIdentityTests(unittest.TestCase):
    def test_metadata_does_not_invalidate_but_backend_source_does(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); backend=root/'backends'/'triton'; backend.mkdir(parents=True)
            (backend/'kernel.py').write_text('source one')
            with patch('bench.run.ROOT',root):
                before=source_identity()
                (backend/'.DS_Store').write_bytes(b'finder metadata')
                (backend/'__pycache__').mkdir(); (backend/'__pycache__'/'kernel.pyc').write_bytes(b'cache')
                self.assertEqual(before['source_tree_sha256'],source_identity()['source_tree_sha256'])
                (backend/'kernel.py').write_text('source two')
                self.assertNotEqual(before['source_tree_sha256'],source_identity()['source_tree_sha256'])
            self.assertEqual(set(before['source_files']),{'backends/triton/kernel.py'})
            self.assertEqual(before['source_identity_version'],2)
