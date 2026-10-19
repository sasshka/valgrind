import gdb


class target_valgrind_command(gdb.Command):
    """target valgrind [args]

    Connect to a local extended-remote target running vgdb, this
    allows GDB to interact with valgrind.

    Any `args` are passed through to vgdb after the --vargs flag.
    """

    def __init__(self):
        gdb.Command.__init__(self, "target valgrind", gdb.COMMAND_NONE)

    def invoke(self, args, from_tty):
        vgdb = "target extended-remote | vgdb --multi"
        if len(args) > 0:
            vgdb += " --vargs " + args
        print(f"Command: {vgdb}")
        try:
            gdb.execute(vgdb, from_tty)
        except gdb.error as error:
            raise gdb.GdbError(f"Could not connect to Valgrind: {error}") from None


target_valgrind_command()
