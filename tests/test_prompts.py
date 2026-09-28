import tempfile
import unittest
from pathlib import Path

from recast.training import read_prompt_records


class PromptRecordTest(unittest.TestCase):
    def test_geneval_jsonl_preserves_structured_metadata(self):
        contents = (
            '{"tag":"counting","include":[{"class":"cat","count":2}],'
            '"prompt":"a photo of two cats"}\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            path.write_text(contents, encoding="utf-8")
            records = read_prompt_records(path)
        self.assertEqual(records[0].prompt, "a photo of two cats")
        self.assertEqual(records[0].metadata["tag"], "counting")
        self.assertEqual(records[0].metadata["include"][0]["count"], 2)


if __name__ == "__main__":
    unittest.main()
