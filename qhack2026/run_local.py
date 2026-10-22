"""Runs the solution against a local copy of the grader rules: python3 run_local.py"""
import pathlib

HERE = pathlib.Path(__file__).parent
code = "\n".join((HERE / f).read_text() for f in ("template_top.py", "solution_part.py", "harness.py"))
exec(compile(code, "personal_best", "exec"), {"__name__": "__main__"})
