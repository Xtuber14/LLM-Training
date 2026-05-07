#!/usr/bin/env python3
"""
Repository source collector.

Study notes: calculator design across multiple languages
========================================================

Building a calculator is a classic logic-heavy project because it touches:
- input parsing
- data types
- error handling
- operator precedence

Below is a compact explanation of how the same idea is commonly expressed in
different languages, with the "expert" version pointing toward expression
evaluation.

1. C++
-------
C++ usually wraps behavior inside a class and relies on exceptions such as
`std::runtime_error` or `std::domain_error` for invalid operations.

Key ideas:
- strong type control with `double`
- manual handling of error cases like division by zero
- a good fit for performance-oriented tools

Example shape:

    class Calculator {
    public:
        double add(double a, double b) { return a + b; }
        double divide(double a, double b) {
            if (b == 0) throw std::runtime_error("Division by zero!");
            return a / b;
        }
    };

2. Python
---------
Python is concise and readable. While `eval()` exists, it should not be used
for untrusted input. Safer beginner-friendly approaches use:
- string parsing
- dictionaries that map operators to functions
- explicit error handling with `try/except`

Example shape:

    self.ops = {
        "+": lambda a, b: a + b,
        "/": self._safe_divide,
    }

3. Java
-------
Java emphasizes structure and explicitness.

Key ideas:
- methods with declared visibility like `public`
- `try/catch` blocks
- exceptions such as `ArithmeticException`
- clear object-oriented design

4. JavaScript
-------------
JavaScript is flexible and especially useful in UI-driven environments.

Key ideas:
- numbers are generally IEEE 754 floating-point values
- `switch` statements are common for operator dispatch
- browser and frontend integration is a natural strength

5. Rust
-------
Rust avoids exceptions in the usual sense and prefers `Result<T, E>`.

Key ideas:
- failures are part of the type system
- callers must explicitly handle success or error
- strong guarantees around correctness and safety

Comparison summary
------------------
- C++: performance and low-level control
- Python: readability and fast iteration
- Java: structure and maintainability
- JavaScript: web integration
- Rust: safety and explicit error handling

Expert challenge: expression evaluation
---------------------------------------
To support an input like:

    3 + 4 * 2 / (1 - 5)^2

you usually need more than simple splitting. A standard next step is the
Shunting-yard algorithm, which:
- reads infix notation
- respects precedence and parentheses
- converts expressions into Reverse Polish Notation (RPN)
- makes evaluation much easier afterward

This file itself is not a calculator; it is a repository file collector. These
notes are kept here as in-code documentation because they may be useful study
material alongside the script.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


LANGUAGE_EXTENSIONS = {
    "c": {".c", ".h"},
    "cpp": {".cpp", ".hpp", ".cc", ".cxx", ".hh", ".hxx", ".h++", ".c++"},
    "csharp": {".cs"},
    "css": {".css", ".scss", ".sass", ".less"},
    "go": {".go"},
    "html": {".html", ".htm"},
    "java": {".java"},
    "python": {".py"},
    "php": {".php", ".phtml"},
    "ruby": {".rb"},
    "rust": {".rs"},
    "sql": {".sql"},
    "javascript": {".js", ".jsx", ".mjs", ".cjs"},
    "typescript": {".ts", ".tsx"},
    "kotlin": {".kt", ".kts"},
    "swift": {".swift"},
    "scala": {".scala"},
    "r": {".r"},
    "matlab": {".m"},
    "perl": {".pl", ".pm"},
    "shell": {".sh", ".bash", ".zsh"},
    "lua": {".lua"},
    "dart": {".dart"},
    "objective-c": {".mm"},
    "elixir": {".ex", ".exs"},
    "haskell": {".hs"},
    "clojure": {".clj", ".cljs", ".cljc"},
    "groovy": {".groovy"},
    "powershell": {".ps1", ".psm1"},
    "visual-basic": {".vb"},
    "assembly": {".asm", ".s"},
    "fortran": {".f", ".f90", ".f95", ".f03", ".f08"},
    "cobol": {".cob", ".cbl"},
    "erlang": {".erl", ".hrl"},
    "julia": {".jl"},
    "ocaml": {".ml", ".mli"},
    "fsharp": {".fs", ".fsi", ".fsx"},
    "nim": {".nim"},
    "zig": {".zig"},
    "solidity": {".sol"},
    "vue": {".vue"},
    "svelte": {".svelte"},
}


def parse_args() -> argparse.Namespace:
    # Parse the repository path and optional output base directory.
    parser = argparse.ArgumentParser(
        description="Copy source files from a repository into data/<language> folders."
    )
    parser.add_argument(
        "repository",
        type=Path,
        help="Path to the repository to scan.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data"),
        help="Output base directory. Defaults to ./data",
    )
    return parser.parse_args()


def should_skip(path: Path, output_dir: Path) -> bool:
    # Skip version-control, dependency, cache, and generated output folders.
    skip_names = {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "dist",
        "build",
    }
    return any(part in skip_names for part in path.parts) or output_dir in path.parents


def language_for_file(path: Path) -> str | None:
    # Match a file extension to the first configured language bucket.
    suffix = path.suffix.lower()
    for language, extensions in LANGUAGE_EXTENSIONS.items():
        if suffix in extensions:
            return language
    return None


def collect_files(repository: Path, output_dir: Path) -> dict[str, int]:
    # Copy matching files while preserving their relative paths to avoid name clashes.
    counts = {language: 0 for language in LANGUAGE_EXTENSIONS}
    repository = repository.resolve()
    output_dir = output_dir.resolve()

    for file_path in repository.rglob("*"):
        if not file_path.is_file():
            continue
        if should_skip(file_path, output_dir):
            continue

        language = language_for_file(file_path)
        if not language:
            continue

        relative_path = file_path.relative_to(repository)
        destination = output_dir / language / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, destination)
        counts[language] += 1

    return counts


def main() -> None:
    # Entry point for command-line usage.
    args = parse_args()
    repository = args.repository
    output_dir = args.output

    if not repository.exists() or not repository.is_dir():
        raise SystemExit(f"Repository path is not a directory: {repository}")

    counts = collect_files(repository, output_dir)

    print(f"Copied files from: {repository.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")
    for language, count in counts.items():
        print(f"{language}: {count}")


if __name__ == "__main__":
    main()
