#!/bin/bash

# Copyright ZGCA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Fast test runner for pre-commit hooks

set -e

echo "🧪 Running pre-commit tests..."

# Check if we're in a git repository
if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "❌ Not in a git repository"
    exit 1
fi

# Get the list of changed Python files
CHANGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$' || true)

if [ -z "$CHANGED_FILES" ]; then
    echo "ℹ️  No Python files changed, skipping tests"
    exit 0
fi

echo "📝 Changed files:"
echo "$CHANGED_FILES"

# Run only fast tests for pre-commit
echo ""
echo "🚀 Running unit tests (fast)..."
python -m pytest tests/ -m "not slow" --maxfail=3 --tb=short -q

# Check if any critical modules were changed
CRITICAL_MODULES="src/pepbenchmark/dataset_loader src/pepbenchmark/pep_utils src/pepbenchmark/metadata.py"

for module in $CRITICAL_MODULES; do
    if echo "$CHANGED_FILES" | grep -q "$module"; then
        echo ""
        echo "⚠️  Critical module $module changed, running targeted tests..."

        # Run specific tests based on changed modules
        if echo "$CHANGED_FILES" | grep -q "src/pepbenchmark/dataset_loader"; then
            python -m pytest tests/test_dataset_loader/ -x --tb=short -q
        fi

        if echo "$CHANGED_FILES" | grep -q "src/pepbenchmark/pep_utils"; then
            python -m pytest tests/test_pep_utils/ -x --tb=short -q
        fi

        if echo "$CHANGED_FILES" | grep -q "src/pepbenchmark/metadata.py"; then
            python -m pytest tests/test_metadata.py -x --tb=short -q
        fi

        break
    fi
done

echo ""
echo "✅ Pre-commit tests passed!"
