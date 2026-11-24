import subprocess
import sys


def test_module_help_identifies_the_cli_and_current_scope():
    result = subprocess.run(
        [sys.executable, "-m", "ricl_experiment", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "RiCL Motivation Study" in result.stdout
    assert "study commands are not yet implemented" in result.stdout
    assert result.stderr == ""
