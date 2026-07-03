#!/usr/bin/env python3

import argparse
import os
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--local',
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        '--headless',
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument('command', nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    run_docker(arguments)


def run_docker(arguments: argparse.Namespace) -> None:
    base = Path(__file__).parent
    command = [
        'podman',
        'run',
        '--rm',
        f'-i{"" if arguments.headless else "t"}',
        '--device',
        'nvidia.com/gpu=all',
        '--shm-size=16g',
        '--userns=keep-id',
        f'--user {os.getgid()}:{os.getuid()}',
        '-v',
        f'{os.getcwd()}:/w:U',
        '-w',
        '/w',
        '--env-file',
        f'{base / ".env"}',
        '--entrypoint',
        '/app/venv/bin/python',
    ]
    if not arguments.local:
        command.extend(
            [
                '--pull=always',
                'ghcr.io/altermarkive/shrubbery:latest',
            ]
        )
    else:
        command.extend(['shrubbery'])
    command.extend(arguments.command[1:])
    command = ' '.join(command)
    Path('workspace/logs').mkdir(parents=True, exist_ok=True)
    command += ' 2>&1 > workspace/logs/$(date +%Y%m%d%H%M%S).log'
    print(command)
    subprocess.run(command, shell=True, check=True)


if __name__ == '__main__':
    main()
