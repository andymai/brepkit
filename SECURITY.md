# Security Policy

## Reporting a Vulnerability

To report a security vulnerability, use [GitHub Security Advisories](https://github.com/andymai/brepkit/security/advisories/new) for private disclosure.

Include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

Expect an initial response within 48 hours.

## Supply Chain

In response to the 2025–2026 wave of npm and GitHub Actions supply-chain
attacks (Shai-Hulud worm, chalk/debug compromise, tj-actions tag retag,
prt-scan AI campaign), the build is configured to fail closed on the
patterns those attacks exploited:

| Defense | Where | What it blocks |
|---|---|---|
| All GitHub Actions pinned to commit SHA | `.github/workflows/*.yml` | Tag-retag attacks (tj-actions class). |
| OSV scan against `Cargo.lock` + `package-lock.json` (PRs report-only, main blocking) | `.github/workflows/osv-scan.yml` | Known-CVE versions in either ecosystem. |
| Dependabot cooldown (7d default / 14d major) across cargo, npm, github-actions | `.github/dependabot.yml` | Fresh malicious uploads. |

Direct install-time cooldown via `.npmrc` `min-release-age` is not enabled
here: npm bundled with Node 24 is 11.6.1, which silently ignores the field
(added in npm 11.10). Add it once Node ships a newer npm.
