import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from qwen_kaggle_health import QwenHealthMonitor


class HealthMonitorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.now = 0.0
        self.output = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.output)
        self.redirect.__enter__()
        self.monitor = QwenHealthMonitor(Path(self.directory.name)/'health.json', interval=86400,
                                         clock=lambda: self.now)
        self.monitor.set_phase('generation', cell=7)

    def tearDown(self):
        self.monitor.close()
        self.redirect.__exit__(None, None, None)
        self.directory.cleanup()

    def test_heartbeat_does_not_hide_stalled_progress(self):
        self.monitor.progress(processed_this_session=10)
        self.now = 301
        self.monitor.emit(force=True)
        self.assertEqual(self.monitor.snapshot()['state'], 'suspected_stall')
        self.assertEqual(self.monitor.snapshot()['progress_age_seconds'], 301)

    def test_identical_callback_does_not_count_as_progress(self):
        self.monitor.progress(processed_this_session=10)
        self.now = 301
        self.monitor.progress(processed_this_session=10, elapsed_seconds=301)
        self.assertEqual(self.monitor.snapshot()['state'], 'suspected_stall')
        self.monitor.progress(processed_this_session=11)
        self.assertEqual(self.monitor.snapshot()['state'], 'running')
        self.assertEqual(self.monitor.snapshot()['progress_age_seconds'], 0)

    def test_private_fields_and_error_messages_are_not_exported(self):
        self.monitor.progress({'processed_this_session': 3, 'title': 'PRIVATE_SENTINEL',
                               'translation': 'PRIVATE_SENTINEL', 'recent_errors': ['PRIVATE_SENTINEL']})
        self.monitor.fail(ValueError('PRIVATE_SENTINEL'))
        stored = self.monitor.path.read_text(encoding='utf-8')
        self.assertNotIn('PRIVATE_SENTINEL', stored + self.output.getvalue())
        self.assertEqual(json.loads(stored)['last_error_class'], 'ValueError')
        self.now = 10000
        self.assertEqual(self.monitor.snapshot()['state'], 'error')

    def test_completion_is_terminal(self):
        self.monitor.cell_finished(final=True)
        self.now = 10000
        self.assertEqual(self.monitor.snapshot()['state'], 'completed')
        self.assertTrue(self.monitor.stop_event.is_set())


if __name__ == '__main__':
    unittest.main()
