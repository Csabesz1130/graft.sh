# Tests on the Linux server

Run `scripts/server-test.sh` from the repository root to check the installed CLI, test suite and demo. For changes to training dependencies, the trainer, or the full release, also run the CPU command in the [acceptance test instructions](README.md#run-the-linux-acceptance-tests). Its `--train` run must pass with zero skipped tests. Report the commit, exit status and any skips for each command.
