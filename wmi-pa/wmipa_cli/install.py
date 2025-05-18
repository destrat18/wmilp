import argparse
import os
import sys

from wmipa_cli.installers.latte import LatteInstaller
from wmipa_cli.installers.symbolic import SymbolicInstaller
from wmipa_cli.installers.volesti import VolestiInstaller
from wmipa_cli.log import logger
from wmipa_cli.utils import safe_cmd


def run():
    args = parse_args(sys.argv[1:])
    if not any((args.all, args.latte, args.volesti, args.symbolic, args.msat, args.nra)):
        print("Nothing to do. Use --help for more information.")
        sys.exit(0)

    if args.all or args.nra:
        install_pysmt_nra(args.force_reinstall)
    elif args.msat:
        install_msat(args.force_reinstall)

    installers = []

    if args.all or args.latte:
        installers.append(LatteInstaller(args.install_path))
    if args.all or args.volesti:
        installers.append(VolestiInstaller(args.install_path))
    if args.all or args.symbolic:
        installers.append(SymbolicInstaller(args.install_path))
    for installer in installers:
        installer.install(args.assume_yes, args.force_reinstall)
    paths_to_export = []
    for installer in installers:
        paths_to_export.extend(installer.paths_to_export)
    if paths_to_export:
        print()
        print("Add the following lines to your ~/.bashrc file:")
        for path in paths_to_export:
            print(f"export PATH={path}:$PATH")
        print()
        print("Then run: source ~/.bashrc")


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--msat", help="Install MathSAT", action="store_true")
    parser.add_argument("--nra", help="Install PySMT version with NRA support", action="store_true")
    parser.add_argument("--latte", help="Install LattE Integrale", action="store_true")
    parser.add_argument("--volesti", help="Install Volesti", action="store_true")
    parser.add_argument("--symbolic", help="Install symbolic integrator (PyXadd)", action="store_true")
    parser.add_argument("--all", help="Install all dependencies", action="store_true")
    parser.add_argument("--install-path", help="Install path for external tools",
                        default=f"{os.path.expanduser('~')}/.wmipa", type=str)
    parser.add_argument("--assume-yes", "-y", help="Automatic yes to prompts", action="store_true")
    parser.add_argument("--force-reinstall", "-f", help="Force reinstallation of dependencies", action="store_true")
    return parser.parse_args(args)


def install_msat(force_reinstall):
    logger.info("Installing MathSAT via pysmt-install...")
    force_str = "--force" if force_reinstall else ""
    safe_cmd(f"pysmt-install --msat --confirm-agreement {force_str}")


def install_pysmt_nra(force_reinstall):
    url = "git+https://git@github.com/masinag/pysmt@nrat#egg=pysmt"
    logger.info(f"Installing PySMT with NRA support from {url}...")
    force_str = "--force-reinstall" if force_reinstall else ""
    safe_cmd(f"{sys.executable} -m pip install {url} {force_str}")
