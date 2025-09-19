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
        '--queue',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--priority',
        type=str,
        default=None,
    )
    parser.add_argument('command', nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    if arguments.queue is None:
        run_docker(arguments)
    else:
        run_kubernetes(arguments)


SPEC_SECRETS = """---
apiVersion: v1
kind: Secret
metadata:
    name: shrubbery-secrets
type: Opaque
stringData:
    NUMERAI_PUBLIC_ID: '{numerai_public_id}'
    NUMERAI_SECRET_KEY: '{numerai_secret_key}'
    WANDB_API_KEY: '{wandb_api_key}'
    WANDB_ENTITY: '{wandb_entity}'
    WANDB_PROJECT: '{wandb_project}'
"""

SPEC_JOB = """---
apiVersion: batch/v1
kind: Job
metadata:
    name: shrubbery-job-{job_id}
    namespace: default
    labels:
        kueue.x-k8s.io/queue-name: {queue_name}
        kueue.x-k8s.io/workload-priority-class: {workload_priority}
spec:
    suspend: true
    template:
        spec:
            containers:
                -
                    name: shrubbery
                    image: {container_image}
                    imagePullPolicy: Always
                    env:
                        -
                            name: HYDRA_FULL_ERROR
                            value: '1'
                        -
                            name: NUMERAI_PUBLIC_ID
                            valueFrom:
                                secretKeyRef:
                                    name: shrubbery-secrets
                                    key: NUMERAI_PUBLIC_ID
                        -
                            name: NUMERAI_SECRET_KEY
                            valueFrom:
                                secretKeyRef:
                                    name: shrubbery-secrets
                                    key: NUMERAI_SECRET_KEY
                        -
                            name: WANDB_API_KEY
                            valueFrom:
                                secretKeyRef:
                                    name: shrubbery-secrets
                                    key: WANDB_API_KEY
                        -
                            name: WANDB_ENTITY
                            valueFrom:
                                secretKeyRef:
                                    name: shrubbery-secrets
                                    key: WANDB_ENTITY
                        -
                            name: WANDB_PROJECT
                            valueFrom:
                                secretKeyRef:
                                    name: shrubbery-secrets
                                    key: WANDB_PROJECT
                    volumeMounts:
                        -
                            name: shrubbery-configs
                            mountPath: /w
                    workingDir: /w
                    resources:
                        limits:
                            nvidia.com/gpu: 1
                    args: {arguments}
            restartPolicy: Never
            volumes:
                -
                    name: shrubbery-configs
                    hostPath:
                        path: /shared/numerai
"""


def run_kubernetes(arguments: argparse.Namespace) -> None:
    configure_secrets()
    enqueue_workload(arguments)


def configure_secrets() -> None:
    base = Path(__file__).parent
    env_path = base / '.env'
    variables = {}
    with env_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip().lower()
            value = value.strip()
            try:
                value = shlex.split(value)[0]
            except Exception:
                pass
            variables[key] = value
        spec_secrets = SPEC_SECRETS.format(**variables)
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as spec_secrets_handle:
            spec_secrets_handle.write(spec_secrets)
            spec_secrets_path = Path(spec_secrets_handle.name)
        os.system(f'kubectl apply -f {spec_secrets_path}')
        spec_secrets_path.unlink()


def enqueue_workload(arguments: argparse.Namespace) -> None:
    spec_workload = SPEC_JOB.format(
        job_id=int(time.time() * 1000),
        queue_name=arguments.queue,
        workload_priority=arguments.priority,
        container_image='shrubbery' if arguments.local else 'ghcr.io/altermarkive/shrubbery:latest',
        arguments=str(arguments.command[1:]),
    )
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as spec_workload_handle:
        spec_workload_handle.write(spec_workload)
        spec_secrets_path = Path(spec_workload_handle.name)
    os.system(f'kubectl create -f {spec_secrets_path}')
    spec_secrets_path.unlink()


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
    print(' '.join(command))
    os.system(' '.join(command))


if __name__ == '__main__':
    main()
