#!/usr/bin/env bash

# Helpful failure messages for pre-commit hooks.

echo "Pre-commit hook failed. Common fixes:"
echo ""
echo "  Formatting:  cargo fmt --all"
echo "  Lint:        cargo clippy --all-targets --fix --allow-dirty"
echo "  Tests:       cargo test --workspace"
echo ""
echo "To skip hooks (emergency): git commit --no-verify"
