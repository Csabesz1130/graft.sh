# Contributing to graft

Bug reports, genomes, verifiers and fixes are welcome.

## Before you open a pull request

- Run the checks: `scripts/server-test.sh` from the repository root. For changes to the trainer or its dependencies,
  also run it with `--train` (see the script's header).
- Keep results real. Numbers in the README and docs come from runs that were actually made; say how you got any you
  add.
- One change per pull request, with a short description of what it fixes and how you checked it.

## The contributor license agreement

graft is licensed under the [FSL-1.1-ALv2](LICENSE), and each version becomes Apache-2.0 after two years. So that
contributions can be offered under those licenses, and under a commercial license, every contributor signs the
[contributor license agreement](CLA.md) once. You keep the copyright in your work.

On your first pull request, a bot asks you to sign by posting this comment:

    I have read the CLA Document and I hereby sign the CLA

The pull request can be merged after that. Your signature is stored in this repository on the `cla-signatures` branch.

## Trademarks

If you publish a modified version, please give it a different name; see [TRADEMARKS.md](TRADEMARKS.md).
