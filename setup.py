import sys

try:
    from setuptools import Extension, setup
except ImportError as exc:  # pragma: no cover - user environment bootstrap path
    raise SystemExit(
        "Missing Python build dependency: setuptools.\n"
        "Bootstrap your Python environment with:\n"
        "  python3 -m ensurepip --upgrade\n"
        "  python3 -m pip install --upgrade pip setuptools wheel\n"
        "Then retry:\n"
        "  python3 setup.py build_ext --inplace"
    ) from exc


def _compile_args():
    if sys.platform.startswith("win"):
        return ["/O2", "/GL"]
    return ["-O3", "-flto"]


def _link_args():
    if sys.platform.startswith("win"):
        return ["/LTCG"]
    return ["-flto"]


setup(
    name="abalone-native",
    version="0.1.0",
    ext_modules=[
        Extension(
            "abalone._native",
            sources=[
                "abalone/_native_src/module.c",
                "abalone/_native_src/tables.c",
                "abalone/_native_src/movegen.c",
                "abalone/_native_src/eval.c",
                "abalone/_native_src/search.c",
            ],
            include_dirs=["abalone/_native_src"],
            extra_compile_args=_compile_args(),
            extra_link_args=_link_args(),
        )
    ],
)
