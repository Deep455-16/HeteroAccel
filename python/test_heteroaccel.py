"""
test_heteroaccel.py - Unit tests for HeteroAccel Python bindings.
Phase 24: Native Streaming and Intelligent Runtime Resource Management.
"""

import os
import sys
import unittest
from pathlib import Path

# Add current directory and parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import heteroaccel as ha


def get_test_model_path():
    env_path = os.environ.get("HETEROACCEL_MODEL_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    default_path = "D:/HeteroAccel/Models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
    if os.path.exists(default_path):
        return default_path
    return None


class TestHeteroAccelSDK(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.version = ha.version()
        except Exception as e:
            raise unittest.SkipTest(f"Failed to load HeteroAccel shared library: {e}")

    def test_version(self):
        v = ha.version()
        self.assertIsInstance(v, str)
        self.assertTrue(v.startswith("0.24"))

    def test_backends(self):
        with ha.Runtime() as rt:
            backends = rt.backends()
            self.assertIsInstance(backends, list)
            self.assertGreaterEqual(len(backends), 1)
            # Check backend fields
            first = backends[0]
            self.assertIn("id", first)
            self.assertIn("available", first)
            self.assertIn("requires_vulkan", first)
            self.assertIn("requires_cuda", first)
            backend_ids = [b["id"] for b in backends]
            self.assertTrue(any("cpu" in bid.lower() for bid in backend_ids))

    def test_model_load_invalid(self):
        with ha.Runtime() as rt:
            with self.assertRaises(ha.HeteroAccelError):
                rt.load_model("this_model_does_not_exist_at_all.gguf")

    def test_perf_profile(self):
        with ha.Runtime() as rt:
            path = rt.perf_path()
            self.assertIsInstance(path, str)
            # Reset should succeed without exception
            rt.perf_reset()

    def test_repeated_cleanup(self):
        # Stress-test creation and destruction of runtimes
        for _ in range(5):
            rt = ha.Runtime()
            backends = rt.backends()
            self.assertGreaterEqual(len(backends), 1)
            del rt

    def test_infer(self):
        model_path = get_test_model_path()
        if not model_path:
            self.skipTest("No test GGUF model available")

        with ha.Runtime() as rt:
            with rt.load_model(model_path) as model:
                self.assertGreater(model.context_length, 0)
                self.assertGreater(model.vocab_size, 0)

                # Check memory estimate
                mem = model.memory_estimate()
                self.assertGreater(mem.weights_bytes, 0)
                self.assertGreater(mem.total_estimated, 0)

                with model.create_session() as session:
                    res = session.infer(prompt="Hello", max_new_tokens=4)
                    self.assertIsInstance(res.output_text, str)
                    self.assertGreater(res.n_output_tokens, 0)
                    self.assertGreater(res.tokens_per_second, 0.0)

    def test_streaming(self):
        model_path = get_test_model_path()
        if not model_path:
            self.skipTest("No test GGUF model available")

        received_tokens = []

        def on_token(piece, token_id):
            received_tokens.append((piece, token_id))
            return 1  # Continue

        with ha.Runtime() as rt:
            with rt.load_model(model_path) as model:
                with model.create_session() as session:
                    res = session.infer_stream(
                        prompt="Once upon a time",
                        callback=on_token,
                        max_new_tokens=5
                    )
                    self.assertGreater(len(received_tokens), 0)
                    self.assertEqual(len(received_tokens), res.n_output_tokens)
                    self.assertGreater(res.tokens_per_second, 0.0)

    def test_cancel(self):
        model_path = get_test_model_path()
        if not model_path:
            self.skipTest("No test GGUF model available")

        tokens_received = []

        def cancel_after_2(piece, token_id):
            tokens_received.append((piece, token_id))
            if len(tokens_received) >= 2:
                return 0  # Signal cancel
            return 1

        with ha.Runtime() as rt:
            with rt.load_model(model_path) as model:
                with model.create_session() as session:
                    res = session.infer_stream(
                        prompt="The quick brown fox",
                        callback=cancel_after_2,
                        max_new_tokens=10
                    )
                    # Inference should have stopped early
                    self.assertLessEqual(len(tokens_received), 3)


if __name__ == "__main__":
    unittest.main()
