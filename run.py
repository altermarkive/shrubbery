#!/usr/bin/env python3

import argparse
import os
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
        '--trace',
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
        'podman',
        'run',
        '--rm',
        f'-i{"" if arguments.headless else "t"}',
        '--shm-size=16g',
        '--userns=keep-id',
        '-v',
        f'{os.getcwd()}:/w:U',
        '-w',
        '/w',
        '--env-file',
        f'{base / ".env"}',
    ]
    if arguments.gpus:
        command.extend(['--device', 'nvidia.com/gpu=all'])
    if not (arguments.lint or arguments.debug):
        command.extend(['--entrypoint', '/usr/local/bin/uv'])
    if arguments.trace:
        command.extend(
            [
                '-e',
                'PROFILE_TRAINING=1',
                '-e',
                'PROFILE_INFERENCE=1',
            ]
        )
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
                '"ruff check --select I src && ruff format --check --diff && ty check --extra-search-path /usr/local/lib/python3.13/dist-packages/"',  # noqa: E501
            ]
        )
    if not arguments.lint and not arguments.debug:
        command.extend(arguments.command[1:])
    command = ' '.join(command)
    if arguments.priority is not None:
        priority = arguments.priority
        command = f'sbatch --wrap="{command}" --priority={priority} --nodes=1 --output=/tmp/slurm-%j.out --error=/tmp/slurm-%j.err'
    print(command)
    os.system(command)


if __name__ == '__main__':
    main()
