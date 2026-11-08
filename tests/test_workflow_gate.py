import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from workflow_gate import recipe_key, source_and_smoke_gate


class WorkflowGateTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((Path(__file__).resolve().parents[1]/"configs/mixed.json").read_text())

    def test_receipt_ignores_smoke_size_and_logging_but_not_learning_configuration(self):
        full = self.cfg
        smoke = copy.deepcopy(full)
        smoke.update(run_id="separate-smoke", test_train_clips=20, test_validation_clips=10,
                     sample_count=1, save_steps=2, eval_steps=2, logging_steps=1)
        self.assertEqual(recipe_key(full,"commit"),recipe_key(smoke,"commit"))
        smoke["learning_rate"] = 1e-4
        self.assertNotEqual(recipe_key(full,"commit"),recipe_key(smoke,"commit"))
        self.assertNotEqual(recipe_key(full,"commit"),recipe_key(full,"new-code"))

    def test_incomplete_source_stops_before_smoke(self):
        info = dict(complete=False, checkpoint="checkpoint-3895",status="paused_for_resume",epoch=.77)
        with patch("workflow_gate.inspect_source",return_value=(info,None)):
            with self.assertRaisesRegex(RuntimeError,"Finish UrduTTS"):
                source_and_smoke_gate(self.cfg,True,"commit",api=object(),download=object())

    def test_full_run_requires_matching_remote_receipt(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["init_source"] = None
        with self.assertRaisesRegex(RuntimeError,"TEST=True"):
            source_and_smoke_gate(cfg,False,"commit",api=object(),download=lambda *a: (_ for _ in ()).throw(FileNotFoundError()))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"receipt.json"
            p.write_text(json.dumps(dict(status="passed",recipe=recipe_key(cfg,"commit"))))
            source_and_smoke_gate(cfg,False,"commit",api=object(),download=lambda *a:p)
            with self.assertRaisesRegex(RuntimeError,"does not match"):
                source_and_smoke_gate(cfg,False,"different",api=object(),download=lambda *a:p)
