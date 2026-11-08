"""Test credential startup before cloning/installing in a fresh Kaggle notebook."""
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class NotebookCredentialTests(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads(Path('kaggle_run.ipynb').read_text())
        tree = ast.parse(''.join(self.notebook['cells'][2]['source']))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_hf_token')
        scope = {'os': os, 'HF_REPO_ID': 'hassan-31x/orpheus-ft-aslp-urdutts'}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<notebook-credentials>', 'exec'), scope)
        self.load = scope['load_hf_token']

    def test_existing_environment_token_does_not_call_kaggle(self):
        env = {'HF_TOKEN': '  test-existing-token  '}
        factory = Mock(side_effect=AssertionError('Kaggle must not be queried'))
        self.assertEqual(self.load(environ=env, secret_client_factory=factory), 'environment:HF_TOKEN')
        self.assertEqual(env['HF_TOKEN'], 'test-existing-token')
        factory.assert_not_called()

    def test_legacy_hub_environment_name_becomes_training_token(self):
        env = {'HUGGING_FACE_HUB_TOKEN': 'test-existing-token'}
        self.load(environ=env, secret_client_factory=Mock(side_effect=AssertionError()))
        self.assertEqual(env['HF_TOKEN'], 'test-existing-token')

    def test_custom_secret_label_and_whitespace_are_supported(self):
        client = Mock()
        client.get_secret.return_value = '  test-kaggle-token\n'
        env = {}
        self.assertEqual(self.load('ASLP_HF_TOKEN', env, lambda: client), 'Kaggle Secrets')
        client.get_secret.assert_called_once_with('ASLP_HF_TOKEN')
        self.assertEqual(env['HF_TOKEN'], 'test-kaggle-token')

    def test_default_secret_import_works_in_fresh_runtime(self):
        client = Mock()
        client.get_secret.return_value = 'test-kaggle-token'
        with patch.dict('sys.modules', {'kaggle_secrets': SimpleNamespace(UserSecretsClient=lambda: client)}):
            env = {}
            self.load(environ=env)
        client.get_secret.assert_called_once_with('HF_TOKEN')
        self.assertEqual(env['HF_TOKEN'], 'test-kaggle-token')

    def test_missing_or_disabled_secret_has_actionable_redacted_error(self):
        factory = Mock(side_effect=RuntimeError('backend echoed test-sensitive-token'))
        env = {}
        with self.assertRaises(RuntimeError) as caught:
            self.load(environ=env, secret_client_factory=factory)
        message = str(caught.exception)
        self.assertIn('Add-ons > Secrets', message)
        self.assertIn('THIS notebook', message)
        self.assertIn('Save Version > Save & Run All', message)
        self.assertNotIn('test-sensitive-token', message)
        self.assertNotIn('HF_TOKEN', env)

    def test_empty_secret_or_label_is_rejected(self):
        for value in ('', '  ', None):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, 'empty'):
                self.load(environ={}, secret_client_factory=lambda: SimpleNamespace(get_secret=lambda name: value))
        with self.assertRaisesRegex(RuntimeError, 'HF_SECRET_NAME'):
            self.load('', environ={})

    def test_notebook_and_config_target_supplied_standalone_repositories(self):
        tree = ast.parse(''.join(self.notebook['cells'][2]['source']))
        settings = {node.targets[0].id: node.value.value for node in tree.body
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Constant)}
        self.assertEqual(settings['REPO_URL'], 'https://github.com/hassan-31x/orpheus-ft-aslp-urdutts.git')
        self.assertEqual(settings['PROJECT_FOLDER'], '.')
        self.assertEqual(settings['HF_REPO_ID'], 'hassan-31x/orpheus-ft-aslp-urdutts')
        cfg = json.loads(Path('configs/mixed.json').read_text())
        self.assertEqual(cfg['hf_repo_id'], settings['HF_REPO_ID'])
        self.assertEqual(cfg['epochs'], 1)
        self.assertIsNone(cfg['init_from_run'])


if __name__ == '__main__':
    unittest.main()
