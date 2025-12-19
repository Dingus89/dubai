import os
import ast
import sys
import importlib.util
from pathlib import Path
from typing import Dict, Set, List, Tuple

# Folders we should ignore
IGNORE_DIRS = {"venv", "node_modules"}
LOG_PATH = "analyze.txt"

ENTRY_POINTS = {"pipeline.py", "main.py", "app.py"}


class FileAnalysis:
    def __init__(self, path: Path):
        self.path = path
        self.imports: Set[str] = set()
        # imports that reference local files
        self.local_imports: Set[str] = set()
        self.functions: Set[str] = set()
        self.classes: Set[str] = set()
        self.calls: Set[str] = set()


def is_hidden(path: Path):
    return any(part.startswith(".") for part in path.parts)


def analyze_python_file(path: Path, project_root: Path) -> FileAnalysis:
    analysis = FileAnalysis(path)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return analysis

    for node in ast.walk(tree):

        # FUNCTIONS
        if isinstance(node, ast.FunctionDef):
            analysis.functions.add(node.name)

        # CLASSES
        elif isinstance(node, ast.ClassDef):
            analysis.classes.add(node.name)

        # IMPORTS
        elif isinstance(node, ast.Import):
            for alias in node.names:
                analysis.imports.add(alias.name.split(".")[0])

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mod = node.module.split(".")[0]
                analysis.imports.add(mod)

                # If the imported module corresponds to a local file
                if (project_root / (mod + ".py")).exists():
                    analysis.local_imports.add(mod)

        # FUNCTION CALLS
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                analysis.calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                analysis.calls.add(node.func.attr)

    return analysis


def recursively_collect_files(project_root: Path) -> Dict[str, FileAnalysis]:
    results: Dict[str, FileAnalysis] = {}

    for root, dirs, files in os.walk(project_root):
        root_path = Path(root)

        # Skip hidden folders and ignored dirs
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".") and d not in IGNORE_DIRS
        ]

        for f in files:
            if f.endswith(".py"):
                path = root_path / f
                rel_path = str(path.relative_to(project_root))
                results[rel_path] = analyze_python_file(path, project_root)

    return results


def detect_entry_points(files: Dict[str, FileAnalysis]) -> List[str]:
    return [f for f in files.keys() if Path(f).name in ENTRY_POINTS]


def resolve_broken_imports(analysis: Dict[str, FileAnalysis], project_root: Path):
    broken_imports = []
    for rel, info in analysis.items():
        for imp in info.imports:
            # Is it a local import?
            local_file = project_root / (imp + ".py")
            if local_file.exists():
                continue

            # Else try importlib
            if importlib.util.find_spec(imp) is None:
                broken_imports.append((rel, imp))
    return broken_imports


def build_call_graph(analysis: Dict[str, FileAnalysis]) -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = {}

    # filename -> set of files it depends on
    for rel, info in analysis.items():
        graph[rel] = set()

        for imp in info.local_imports:
            target = imp + ".py"
            if target in analysis:
                graph[rel].add(target)

    return graph


def bfs_reachable(start_files: List[str], graph: Dict[str, Set[str]]):
    visited = set()
    stack = list(start_files)

    while stack:
        f = stack.pop()
        if f in visited:
            continue
        visited.add(f)
        for nxt in graph.get(f, []):
            if nxt not in visited:
                stack.append(nxt)
    return visited


def generate_log(
    analysis: Dict[str, FileAnalysis],
    entry_points: List[str],
    reachable: Set[str],
    broken_imports: List[Tuple[str, str]],
    project_root: Path
):
    unused_files = set(analysis.keys()) - reachable

    with open(LOG_PATH, "w", encoding="utf-8") as f:

        f.write("===== PROJECT ANALYSIS REPORT =====\n\n")
        f.write(f"Project Root: {project_root}\n\n")

        # ENTRY POINTS
        f.write("#  ## Entry Points Detected:\n")
        if entry_points:
            for e in entry_points:
                f.write(f"  - {e}\n")
        else:
            f.write("  (None found)\n")
        f.write("\n\n")

        # BROKEN IMPORTS
        f.write("#  ## Broken Imports:\n")
        if broken_imports:
            for file, imp in broken_imports:
                f.write(f"  {file} → {imp}\n")
        else:
            f.write("  (None)\n")
        f.write("\n\n")

        # UNUSED FILES
        f.write("#  ## Unused Files (not reachable from entry points):\n")
        for u in sorted(unused_files):
            f.write(f"  - {u}\n")
        f.write("\n\n")

        # PER-FILE SUMMARY
        f.write("#  ## File Summary:\n")
        for rel, info in analysis.items():
            f.write(f"\nFILE: {rel}\n")
            f.write("  Imports: " + ", ".join(sorted(info.imports)) + "\n")
            f.write("  Local Imports: " +
                    ", ".join(sorted(info.local_imports)) + "\n")
            f.write("  Functions: " + ", ".join(sorted(info.functions)) + "\n")
            f.write("  Classes: " + ", ".join(sorted(info.classes)) + "\n")
            f.write("  Calls: " + ", ".join(sorted(info.calls)) + "\n")
            f.write("\n")

    print(f"\n✔ Analysis complete. Log written to: {LOG_PATH}\n")


def main():
    project_root = Path(os.getcwd())

    print(f"Analyzing project at: {project_root}\n")

    # Scan files
    analysis = recursively_collect_files(project_root)

    # Find entry points
    entry_points = detect_entry_points(analysis)

    # Build dependency graph (import-level)
    graph = build_call_graph(analysis)

    # Traverse from entry points
    reachable = bfs_reachable(entry_points, graph)

    # Find broken imports
    broken_imports = resolve_broken_imports(analysis, project_root)

    # Generate log
    generate_log(analysis, entry_points, reachable,
                 broken_imports, project_root)


if __name__ == "__main__":
    main()
