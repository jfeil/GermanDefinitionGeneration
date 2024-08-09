import subprocess

import click

from src.ha_utils import set_sensor_state, Input


def execute_bash_line(line):
    """Executes a single line of a bash script."""
    process = subprocess.Popen(line, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Print stdout line by line as it is generated
    for stdout_line in process.stdout:
        print(stdout_line.decode(), end='')

    # Wait for the process to finish and get the exit code
    process.wait()

    if process.returncode != 0:
        stderr = process.stderr.read()
        print(f"Error executing line: {line}")
        print(f"Error: {stderr.strip()}")

@click.command()
@click.argument('script_path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
def main(script_path):
    """Executes a bash script line by line with the possibility to add custom code between each line."""
    with open(script_path, 'r') as file:
        lines = file.readlines()
    lines = list(filter(lambda x: not x.startswith("#") or not x, lines))
    set_sensor_state(0, len(lines), Input.TOTAL_INPUT)
    for i, line in enumerate(lines):
        # Strip the line to avoid issues with leading/trailing spaces and newlines
        line = line.strip()

        if line and not line.startswith("#"):  # Skip empty lines and comments
            set_sensor_state(i, len(lines), Input.TOTAL_INPUT)
            print(f"Executing bash line: {line}")
            execute_bash_line(line)

            # Here you can add custom Python code between bash commands
    set_sensor_state(len(lines), len(lines), Input.TOTAL_INPUT)


if __name__ == "__main__":
    main()
