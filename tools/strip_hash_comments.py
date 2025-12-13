import argparse
import os
import sys
from io import StringIO
import tokenize


def strip_hash_comments_from_code(code: str) -> tuple[str, int]:
    """
    Remove Python hash comments while preserving shebangs and docstrings.

    Returns (new_code, removed_count).
    """
    removed = 0
    out_tokens = []
    sio = StringIO(code)
    try:
        tokens = list(tokenize.generate_tokens(sio.readline))
    except tokenize.TokenError:

        return code, 0

    for tok in tokens:
        tok_type, tok_str, start, end, line = tok
        if tok_type == tokenize.COMMENT:

            if start[0] == 1 and tok_str.startswith('#!'):
                out_tokens.append(tok)
            else:
                removed += 1
                continue
        else:
            out_tokens.append(tok)

    new_code = tokenize.untokenize(out_tokens)


    new_code = "\n".join([ln.rstrip() for ln in new_code.splitlines()]) + ("\n" if new_code.endswith("\n") or code.endswith("\n") else "")

    return new_code, removed


def should_skip_path(path: str) -> bool:

    parts = set(os.path.normpath(path).split(os.sep))
    skip_dirs = {'.git', '.venv', 'venv', '.mypy_cache', '.pytest_cache', '__pycache__', 'node_modules'}
    return any(p in skip_dirs for p in parts)


def process_file(path: str) -> tuple[bool, int]:
    try:
        with tokenize.open(path) as f:
            src = f.read()
    except OSError:
        return False, 0

    new_src, removed = strip_hash_comments_from_code(src)
    if removed > 0 and new_src != src:
        with open(path, 'w', encoding='utf-8', newline='') as f:
            f.write(new_src)
        return True, removed
    return False, 0


def main() -> int:
    parser = argparse.ArgumentParser(description='Strip # comments from Python files.')
    parser.add_argument('paths', nargs='*', default=['.'], help='Paths to scan (default: current directory)')
    args = parser.parse_args()

    changed_files = 0
    removed_total = 0

    for base in args.paths:
        if os.path.isfile(base):
            if base.endswith('.py') and not should_skip_path(base):
                changed, removed = process_file(base)
                if changed:
                    print(f"updated {base} (-{removed} comments)")
                    changed_files += 1
                    removed_total += removed
            continue

        for root, dirs, files in os.walk(base):

            dirs[:] = [d for d in dirs if not should_skip_path(os.path.join(root, d))]
            for name in files:
                if not name.endswith('.py'):
                    continue
                path = os.path.join(root, name)
                if should_skip_path(path):
                    continue
                changed, removed = process_file(path)
                if changed:
                    print(f"updated {path} (-{removed} comments)")
                    changed_files += 1
                    removed_total += removed

    print(f"done: {changed_files} file(s) updated, {removed_total} comment(s) removed")
    return 0


if __name__ == '__main__':
    sys.exit(main())

