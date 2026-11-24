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
