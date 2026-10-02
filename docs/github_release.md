# GitHub publication checklist

The existing remote is `https://github.com/BrunsonLiu/RL_FJSP.git`.
Publishing here means sharing a research code repository, not deploying a web
service. No PyPI publication or hosted application is configured.

## Prepare locally

1. Run the checks listed in `CONTRIBUTING.md`.
2. Review `git status --short` and `git diff --cached` as well as unstaged changes.
3. Include the new improvement environment and related model modules together:
   `fjsp.env` currently imports them, so a partial commit can break imports.
4. Review tracked files and Git history for private data and credentials.
5. Confirm benchmark redistribution terms and choose a license for your own
   code. No license is automatically assigned by this engineering update.
6. Verify literature references before publishing SOTA/optimality claims.

## Publish

Authenticate on the local machine, without placing a token in project files:

```shell
gh auth login
gh auth setup-git
git remote -v
```

Use a dedicated branch and review the exact snapshot before committing. Avoid
force pushes. If the remote has newer work, fetch and inspect the difference
before merging it. Publish the reviewed branch and open a pull request:

```shell
git push -u origin HEAD
gh pr create --base master
```

The target branch must be confirmed from the actual GitHub repository; `master`
above reflects the current local branch, not a guaranteed remote default.
After pushing, check the Actions runs and require CI before merging. Local
Windows tests do not replace the Linux and Python 3.10 CI checks.

## Scope of the package

The wheel contains `fjsp` and `rl` modules. Benchmark files, paper sources,
scripts, and experimental outputs remain in the source checkout. Existing
training entry points use checkout-relative defaults, so run the documented
commands from a clone with an editable install. Installed wheels can be used
as a library; pass explicit instance/output paths when using module entry points.

## Known result discrepancy

The publication check on 2026-10-03 validated all five curated schedule files.
For MK12, `sota_final.json` reports `final_best=508`, whereas the saved
`sota_mk12_schedule.json` validates at makespan 524. The schedule is legal but
does not substantiate the table's 508 result. Both original artifacts are
preserved. Locate and independently validate a 508 schedule before treating
that table entry as reproduced. The curated schedule CI check verifies legality,
not agreement with every reported metric or literature optimality.
