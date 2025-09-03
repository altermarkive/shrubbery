#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path


def main() -> None:
    base = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpus',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        '--lint',
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        '--debug',
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        '--local',
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument('command', nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = [
        'docker',
        'run',
        '--rm',
        '-it',
        '-v',
        f'{os.getcwd()}:/w',
        '-w',
        '/w',
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
    print(' '.join(command))
    os.system(' '.join(command))


if __name__ == '__main__':
    main()
