"""Get drone namespaces from AS2 world config file."""

__authors__ = 'Based on aerostack2/project_as2_multirotor_simulator'
__license__ = 'BSD-3-Clause'

import argparse
from pathlib import Path

import yaml


def get_drones_namespaces(filename: Path) -> list[str]:
    """
    Get drone namespaces listed in world config file.

    :param filename: Path to world config YAML file
    :return: List of drone namespaces
    """
    with open(filename, 'r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)

    drones_namespaces = []
    for key in config:
        if key == '/**':
            continue
        drones_namespaces.append(key)

    if len(drones_namespaces) == 0:
        raise ValueError('No drones found in config file')

    return drones_namespaces


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Get drone namespaces from world config file')
    parser.add_argument(
        '-p', '--config_file',
        type=str,
        required=True,
        help='Path to world config YAML file')
    parser.add_argument(
        '--sep',
        type=str,
        default=',',
        help='Separator for output namespaces')

    args = parser.parse_args()
    config_file = Path(args.config_file)

    if not config_file.exists():
        raise FileNotFoundError(f'File {config_file} not found')

    drones = get_drones_namespaces(config_file)
    print(args.sep.join(drones))
