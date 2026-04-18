from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ScriptLauncherTest(unittest.TestCase):
    def test_image_embedding_server_uses_siglip2_pooling(self) -> None:
        script = (ROOT / "scripts" / "run_image_embed_server.sh").read_text(encoding="utf-8")

        self.assertIn("google/siglip2-giant-opt-patch16-384", script)
        self.assertIn("--runner", script)
        self.assertNotIn("Qwen/Qwen3-VL-Embedding-2B", script)

    def test_text_embedding_server_uses_e5_base_v2(self) -> None:
        script = (ROOT / "scripts" / "run_text_embed_server.sh").read_text(encoding="utf-8")

        self.assertIn("intfloat/e5-base-v2", script)
        self.assertIn("PORT=\"${PORT:-8006}\"", script)
        self.assertIn("MODEL=\"${MODEL:-intfloat/e5-base-v2}\"", script)


if __name__ == "__main__":
    unittest.main()
