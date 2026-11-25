import subprocess
import sys


def test_module_help_lists_single_sample_run_and_resume_commands():
    result = subprocess.run(
        [sys.executable, "-m", "ricl_experiment", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "RiCL Motivation Study" in result.stdout
    assert "run" in result.stdout
    assert "resume" in result.stdout
    assert "ms3-diagnose" in result.stdout
    assert "ms3-analyze" in result.stdout
    assert "ms3-repair" in result.stdout
    assert "analyze-ms3-repair" in result.stdout
    assert "ms3-pilot-check" in result.stdout
    assert "ms2-analyze" in result.stdout
    assert result.stderr == ""


def test_module_help_lists_ms1_offline_commands():
    result = subprocess.run(
        [sys.executable, "-m", "ricl_experiment", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    for command in (
        "ms1-plan",
        "ms1-blind",
        "ms1-import-annotations",
        "ms1-analyze",
        "ms1-export-cases",
    ):
        assert command in result.stdout
