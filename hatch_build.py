from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sysconfig
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        # Keep installs working everywhere by default.
        if os.environ.get("VISDET_BUILD_CPP_EXT", "0") != "1":
            return

        # Only build compiled artifacts for wheels.
        if self.target_name != "wheel":
            return

        root = Path(self.root)
        build_tmp = root / ".hatch-ext" / "cpp"
        build_tmp.mkdir(parents=True, exist_ok=True)

        src = build_tmp / "demo_ext.c"
        src.write_text(
            """
#include <Python.h>

static PyObject* demo_add(PyObject* self, PyObject* args) {
    long a;
    long b;
    if (!PyArg_ParseTuple(args, \"ll\", &a, &b)) {
        return NULL;
    }
    return PyLong_FromLong(a + b);
}

static PyMethodDef DemoMethods[] = {
    {\"add\", demo_add, METH_VARARGS, \"Add two integers.\"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef demomodule = {
    PyModuleDef_HEAD_INIT,
    \"_demo_ext\",
    \"visdet demo C extension\",
    -1,
    DemoMethods
};

PyMODINIT_FUNC PyInit__demo_ext(void) {
    return PyModule_Create(&demomodule);
}
""".lstrip(),
            encoding="utf-8",
        )

        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
        if not ext_suffix:
            self.app.display_warning("Could not determine EXT_SUFFIX; skipping C extension build")
            return

        out = build_tmp / f"_demo_ext{ext_suffix}"

        include_dir = sysconfig.get_path("include")
        ldshared = sysconfig.get_config_var("LDSHARED") or "cc -shared"
        cflags = sysconfig.get_config_var("CFLAGS") or ""
        cppflags = sysconfig.get_config_var("CPPFLAGS") or ""

        cmd = (
            shlex.split(ldshared)
            + shlex.split(cflags)
            + shlex.split(cppflags)
            + [f"-I{include_dir}", "-o", str(out), str(src)]
        )

        self.app.display_info(f"Building C extension: {out.name}")
        subprocess.check_call(cmd, cwd=str(root))

        # Ensure hatchling includes the generated artifact in the wheel (without polluting the source tree).
        build_data.setdefault("force_include", {})[str(out)] = f"visdet/_ext_demo/{out.name}"

        # Avoid accidentally shipping .dSYM bundles (macOS).
        dsym = out.with_suffix(out.suffix + ".dSYM")
        if dsym.exists():
            shutil.rmtree(dsym)
