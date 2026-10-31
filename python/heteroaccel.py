"""
HeteroAccel Python bindings via ctypes.
Phase 24: Native Streaming and Intelligent Runtime Resource Management.
Supports Python 3.8+ on Windows and Linux.
"""

import ctypes
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

# Handle types
HaRuntime = ctypes.c_void_p
HaModel = ctypes.c_void_p
HaSession = ctypes.c_void_p

# Error enum codes
HA_OK = 0
HA_ERR_INVALID_ARG = 1
HA_ERR_MODEL_LOAD = 2
HA_ERR_NO_BACKEND = 3
HA_ERR_INFERENCE = 4
HA_ERR_CANCELLED = 5
HA_ERR_OOM = 6
HA_ERR_UNSUPPORTED = 7
HA_ERR_INTERNAL = 8

ERROR_NAMES = {
    HA_OK: "HA_OK",
    HA_ERR_INVALID_ARG: "HA_ERR_INVALID_ARG",
    HA_ERR_MODEL_LOAD: "HA_ERR_MODEL_LOAD",
    HA_ERR_NO_BACKEND: "HA_ERR_NO_BACKEND",
    HA_ERR_INFERENCE: "HA_ERR_INFERENCE",
    HA_ERR_CANCELLED: "HA_ERR_CANCELLED",
    HA_ERR_OOM: "HA_ERR_OOM",
    HA_ERR_UNSUPPORTED: "HA_ERR_UNSUPPORTED",
    HA_ERR_INTERNAL: "HA_ERR_INTERNAL",
}

class HeteroAccelError(Exception):
    def __init__(self, code: int, message: str = ""):
        self.code = code
        self.code_name = ERROR_NAMES.get(code, f"UNKNOWN_{code}")
        msg = message if message else self.code_name
        super().__init__(f"{self.code_name} ({code}): {msg}")

class HaInferParams(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("token_ids", ctypes.POINTER(ctypes.c_int)),
        ("n_token_ids", ctypes.c_int),
        ("max_new_tokens", ctypes.c_int),
        ("temperature", ctypes.c_float),
        ("top_p", ctypes.c_float),
    ]

class HaResult(ctypes.Structure):
    _fields_ = [
        ("output_text", ctypes.c_char_p),
        ("output_token_ids", ctypes.POINTER(ctypes.c_int)),
        ("n_output_tokens", ctypes.c_int),
        ("n_prompt_tokens", ctypes.c_int),
        ("tokens_per_second", ctypes.c_float),
        ("gpu_used", ctypes.c_int),
        ("backend_used", ctypes.c_char_p),
        ("error", ctypes.c_int),
        ("error_message", ctypes.c_char_p),
    ]

class HaBackendInfo(ctypes.Structure):
    _fields_ = [
        ("backend_id", ctypes.c_char_p),
        ("available", ctypes.c_int),
        ("requires_vulkan", ctypes.c_int),
        ("requires_cuda", ctypes.c_int),
    ]

class HaMemoryEstimate(ctypes.Structure):
    _fields_ = [
        ("weights_bytes", ctypes.c_size_t),
        ("kv_cache_bytes", ctypes.c_size_t),
        ("activations_bytes", ctypes.c_size_t),
        ("overhead_bytes", ctypes.c_size_t),
        ("total_estimated", ctypes.c_size_t),
        ("unified_memory", ctypes.c_int),
        ("notes", ctypes.c_char_p),
    ]

HaTokenCallback = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_void_p
)

_lib = None

def _load_lib():
    global _lib
    if _lib is not None:
        return _lib

    base = Path(__file__).resolve().parent.parent
    dll_name = "heteroaccel.dll" if sys.platform == "win32" else "libheteroaccel.so"

    candidates = []
    env_path = os.environ.get("HETEROACCEL_LIB_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([
        base / "build" / "Release" / dll_name,
        base / "build" / "bin" / "Release" / dll_name,
        base / "build" / "bin" / dll_name,
        base / "build" / dll_name,
        base / "bin" / dll_name,
        Path(sys.prefix) / "bin" / dll_name,
        dll_name,
    ])

    for c in candidates:
        try:
            if isinstance(c, Path) and not c.exists():
                continue
            _lib = ctypes.CDLL(str(c))
            break
        except OSError:
            continue

    if _lib is None:
        raise RuntimeError(f"Cannot find HeteroAccel shared library ({dll_name})")

    _setup_signatures(_lib)
    return _lib

def _setup_signatures(lib):
    # Runtime
    lib.ha_runtime_create.argtypes = []
    lib.ha_runtime_create.restype = HaRuntime

    lib.ha_runtime_destroy.argtypes = [HaRuntime]
    lib.ha_runtime_destroy.restype = None

    lib.ha_version_string.argtypes = []
    lib.ha_version_string.restype = ctypes.c_char_p

    lib.ha_runtime_perf_reset.argtypes = [HaRuntime]
    lib.ha_runtime_perf_reset.restype = ctypes.c_int

    lib.ha_runtime_perf_path.argtypes = [HaRuntime, ctypes.c_char_p, ctypes.c_size_t]
    lib.ha_runtime_perf_path.restype = ctypes.c_int

    # Model
    lib.ha_model_load.argtypes = [HaRuntime, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
    lib.ha_model_load.restype = HaModel

    lib.ha_model_unload.argtypes = [HaModel]
    lib.ha_model_unload.restype = None

    lib.ha_model_arch.argtypes = [HaModel]
    lib.ha_model_arch.restype = ctypes.c_char_p

    lib.ha_model_context_length.argtypes = [HaModel]
    lib.ha_model_context_length.restype = ctypes.c_uint32

    lib.ha_model_vocab_size.argtypes = [HaModel]
    lib.ha_model_vocab_size.restype = ctypes.c_uint32

    lib.ha_model_memory_estimate.argtypes = [HaModel, ctypes.c_uint32, ctypes.POINTER(HaMemoryEstimate)]
    lib.ha_model_memory_estimate.restype = ctypes.c_int

    # Session
    lib.ha_session_create.argtypes = [HaModel, ctypes.POINTER(ctypes.c_int)]
    lib.ha_session_create.restype = HaSession

    lib.ha_session_destroy.argtypes = [HaSession]
    lib.ha_session_destroy.restype = None

    lib.ha_session_cancel.argtypes = [HaSession]
    lib.ha_session_cancel.restype = ctypes.c_int

    lib.ha_infer_cancel.argtypes = [HaSession]
    lib.ha_infer_cancel.restype = ctypes.c_int

    lib.ha_infer.argtypes = [HaSession, ctypes.POINTER(HaInferParams)]
    lib.ha_infer.restype = HaResult

    lib.ha_infer_stream.argtypes = [HaSession, ctypes.POINTER(HaInferParams), HaTokenCallback, ctypes.c_void_p]
    lib.ha_infer_stream.restype = ctypes.c_int

    # Backends
    lib.ha_backend_count.argtypes = [HaRuntime]
    lib.ha_backend_count.restype = ctypes.c_int

    lib.ha_backend_get.argtypes = [HaRuntime, ctypes.c_int, ctypes.POINTER(HaBackendInfo)]
    lib.ha_backend_get.restype = ctypes.c_int

    lib.ha_error_string.argtypes = [ctypes.c_int]
    lib.ha_error_string.restype = ctypes.c_char_p


class Runtime:
    def __init__(self):
        lib = _load_lib()
        self._handle = lib.ha_runtime_create()
        if not self._handle:
            raise HeteroAccelError(HA_ERR_INTERNAL, "Failed to create runtime")

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if hasattr(self, "_handle") and self._handle:
            _load_lib().ha_runtime_destroy(self._handle)
            self._handle = None

    @property
    def handle(self):
        return self._handle

    @staticmethod
    def version() -> str:
        lib = _load_lib()
        s = lib.ha_version_string()
        return s.decode("utf-8") if s else ""

    def backends(self) -> List[Dict[str, Union[str, bool]]]:
        lib = _load_lib()
        count = lib.ha_backend_count(self._handle)
        result = []
        for i in range(count):
            info = HaBackendInfo()
            err = lib.ha_backend_get(self._handle, i, ctypes.byref(info))
            if err == HA_OK:
                b_id = info.backend_id.decode("utf-8") if info.backend_id else ""
                result.append({
                    "id": b_id,
                    "available": bool(info.available),
                    "requires_vulkan": bool(info.requires_vulkan),
                    "requires_cuda": bool(info.requires_cuda),
                })
        return result

    def perf_reset(self):
        lib = _load_lib()
        err = lib.ha_runtime_perf_reset(self._handle)
        if err != HA_OK:
            raise HeteroAccelError(err, "Failed to reset perf profile")

    def perf_path(self) -> str:
        lib = _load_lib()
        buf = ctypes.create_string_buffer(1024)
        err = lib.ha_runtime_perf_path(self._handle, buf, len(buf))
        if err != HA_OK:
            raise HeteroAccelError(err, "Failed to get perf profile path")
        return buf.value.decode("utf-8")

    def load_model(self, path: str) -> "Model":
        return Model(self, path)


class Model:
    def __init__(self, runtime: Runtime, path: str):
        self._runtime = runtime
        lib = _load_lib()
        err_out = ctypes.c_int(0)
        c_path = path.encode("utf-8")
        self._handle = lib.ha_model_load(runtime.handle, c_path, ctypes.byref(err_out))
        if not self._handle or err_out.value != HA_OK:
            err_msg = lib.ha_error_string(err_out.value).decode("utf-8")
            raise HeteroAccelError(err_out.value, f"Failed to load model '{path}': {err_msg}")

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if hasattr(self, "_handle") and self._handle:
            _load_lib().ha_model_unload(self._handle)
            self._handle = None

    @property
    def handle(self):
        return self._handle

    def arch(self) -> str:
        lib = _load_lib()
        a = lib.ha_model_arch(self._handle)
        return a.decode("utf-8") if a else ""

    def context_length(self) -> int:
        lib = _load_lib()
        return int(lib.ha_model_context_length(self._handle))

    def vocab_size(self) -> int:
        lib = _load_lib()
        return int(lib.ha_model_vocab_size(self._handle))

    def memory_estimate(self, max_seq_len: int = 2048) -> Dict[str, Union[int, bool, str]]:
        lib = _load_lib()
        est = HaMemoryEstimate()
        err = lib.ha_model_memory_estimate(self._handle, max_seq_len, ctypes.byref(est))
        if err != HA_OK:
            raise HeteroAccelError(err, "Failed to compute model memory estimate")
        return {
            "weights_bytes": int(est.weights_bytes),
            "kv_cache_bytes": int(est.kv_cache_bytes),
            "activations_bytes": int(est.activations_bytes),
            "overhead_bytes": int(est.overhead_bytes),
            "total_estimated": int(est.total_estimated),
            "unified_memory": bool(est.unified_memory),
            "notes": est.notes.decode("utf-8") if est.notes else "",
        }

    def create_session(self) -> "Session":
        return Session(self)


class Session:
    def __init__(self, model: Model):
        self._model = model
        lib = _load_lib()
        err_out = ctypes.c_int(0)
        self._handle = lib.ha_session_create(model.handle, ctypes.byref(err_out))
        if not self._handle or err_out.value != HA_OK:
            err_msg = lib.ha_error_string(err_out.value).decode("utf-8")
            raise HeteroAccelError(err_out.value, f"Failed to create session: {err_msg}")

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if hasattr(self, "_handle") and self._handle:
            _load_lib().ha_session_destroy(self._handle)
            self._handle = None

    @property
    def handle(self):
        return self._handle

    def infer(self, prompt: Optional[str] = None,
              token_ids: Optional[List[int]] = None,
              max_tokens: int = 16,
              temperature: float = 0.0,
              top_p: float = 0.95) -> Dict[str, Union[str, int, float, bool, List[int]]]:
        lib = _load_lib()
        params = HaInferParams()
        params.max_new_tokens = max_tokens
        params.temperature = temperature
        params.top_p = top_p

        c_prompt = None
        if prompt is not None:
            c_prompt = prompt.encode("utf-8")
            params.prompt = c_prompt
        else:
            params.prompt = None

        c_token_ids = None
        if token_ids is not None and len(token_ids) > 0:
            c_token_ids = (ctypes.c_int * len(token_ids))(*token_ids)
            params.token_ids = c_token_ids
            params.n_token_ids = len(token_ids)
        else:
            params.token_ids = None
            params.n_token_ids = 0

        res = lib.ha_infer(self._handle, ctypes.byref(params))
        if res.error != HA_OK:
            msg = res.error_message.decode("utf-8") if res.error_message else "Inference failed"
            raise HeteroAccelError(res.error, msg)

        out_text = res.output_text.decode("utf-8") if res.output_text else ""
        out_ids = []
        if res.output_token_ids and res.n_output_tokens > 0:
            out_ids = [res.output_token_ids[i] for i in range(res.n_output_tokens)]

        return {
            "output_text": out_text,
            "token_ids": out_ids,
            "n_output_tokens": int(res.n_output_tokens),
            "n_prompt_tokens": int(res.n_prompt_tokens),
            "tokens_per_second": float(res.tokens_per_second),
            "gpu_used": bool(res.gpu_used),
            "backend_used": res.backend_used.decode("utf-8") if res.backend_used else "",
        }

    def infer_stream(self, prompt: Optional[str] = None,
                     callback: Optional[Callable[[str, int], bool]] = None,
                     token_ids: Optional[List[int]] = None,
                     max_tokens: int = 16,
                     temperature: float = 0.0,
                     top_p: float = 0.95):
        """
        Runs native token streaming.
        callback(token_text: str, token_id: int) -> bool
        Return False to cancel generation.
        """
        lib = _load_lib()
        params = HaInferParams()
        params.max_new_tokens = max_tokens
        params.temperature = temperature
        params.top_p = top_p

        c_prompt = None
        if prompt is not None:
            c_prompt = prompt.encode("utf-8")
            params.prompt = c_prompt
        else:
            params.prompt = None

        c_token_ids = None
        if token_ids is not None and len(token_ids) > 0:
            c_token_ids = (ctypes.c_int * len(token_ids))(*token_ids)
            params.token_ids = c_token_ids
            params.n_token_ids = len(token_ids)
        else:
            params.token_ids = None
            params.n_token_ids = 0

        def c_callback(text_ptr, tid, user_data):
            if callback is None:
                return 0
            text = text_ptr.decode("utf-8") if text_ptr else ""
            keep_going = callback(text, int(tid))
            return 0 if keep_going else 1

        cb_thunk = HaTokenCallback(c_callback)
        err = lib.ha_infer_stream(self._handle, ctypes.byref(params), cb_thunk, None)
        if err != HA_OK and err != HA_ERR_CANCELLED:
            err_msg = lib.ha_error_string(err).decode("utf-8")
            raise HeteroAccelError(err, f"Streaming failed: {err_msg}")
        return err

    def cancel(self):
        lib = _load_lib()
        lib.ha_session_cancel(self._handle)
