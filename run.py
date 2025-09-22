#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shlex
import tempfile
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpus',
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        '--lint',
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        '--debug',
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        '--local',
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        '--headless',
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        '--priority',
        type=str,
        default=None,
    )
    parser.add_argument('command', nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    run_docker(arguments)


def run_docker(arguments: argparse.Namespace) -> None:
    base = Path(__file__).parent
    command = [
        'docker',
        'run',
        '--rm',
        f'-i{"" if arguments.headless else "t"}',
        '-v',
        f'{os.getcwd()}:/w',
        '-w',
        '/w',
        '--env',
        'HYDRA_FULL_ERROR=1',
        '--env-file',
        f'{base / ".env"}',
    ]
    if arguments.gpus:
        command.extend(['--gpus', 'all'])
    if arguments.lint or arguments.debug:
        command.extend(['--entrypoint', '/bin/bash'])
    if not arguments.local:
        command.extend(
            [
                '--pull=always',
                'ghcr.io/altermarkive/shrubbery:latest',
            ]
        )
    else:
        command.extend(['shrubbery'])
    if arguments.lint:
        command.extend(
            [
                '-c',
                '"ruff check --select I src; ruff format --check; mypy src"',  # noqa: E501
            ]
        )
    if not arguments.lint and not arguments.debug:
        command.extend(arguments.command[1:])
    command = ' '.join(command)
    if arguments.priority is not None:
        priority = arguments.priority
        command = f'sbatch --wrap="{command}" --priority={priority} --nodes=1 --output=/tmp/slurm-%j.out'
    print(command)
    os.system(command)


if __name__ == '__main__':
    main()
