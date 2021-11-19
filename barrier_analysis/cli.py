from argparse import ArgumentParser
from barrier_analysis import __version__
from barrier_analysis import dashboard

import panel as pn

def dashboard1(args):
    print('Do something with subcommand 1', args)
    pn.serve(dashboard.create_dashboard(data_dir=args.data_dir))

def cli(args=None):
    p = ArgumentParser(
        description="Barrier Impact Analysis ",
        conflict_handler='resolve'
    )
    p.set_defaults(func=lambda args: p.print_help())
    p.add_argument(
        '-V', '--version',
        action='version',
        help='Show the conda-prefix-replacement version number and exit.',
        version="barrier_analysis %s" % __version__,
    )

    args, unknown = p.parse_known_args(args)

    # do something with the sub commands
    sub_p = p.add_subparsers(help='sub-command help')
    # add show all sensors command
    subcmd1 = sub_p.add_parser('dash1', help='dashboard 1')
    subcmd1.add_argument('--data-dir', type=str, required=False, help='the directory containing all the files')
    subcmd1.set_defaults(func=dashboard1)

    # Now call the appropriate response.
    pargs = p.parse_args(args)
    pargs.func(pargs)
    return 
    # No return value means no error.
    # Return a value of 1 or higher to signify an error.
    # See https://docs.python.org/3/library/sys.html#sys.exit


if __name__ == '__main__':
    import sys
    cli(sys.argv[1:])
