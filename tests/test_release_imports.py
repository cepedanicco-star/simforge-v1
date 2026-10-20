"""The source distribution must include every internally imported module."""
import ast
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_internal_import_closure():
    missing=[]
    for base in ('pipeline','training'):
        for path in (ROOT/base).rglob('*.py'):
            parts=list(path.relative_to(ROOT).with_suffix('').parts)
            package=parts[:-1]
            for node in ast.walk(ast.parse(path.read_text(encoding='utf8'))):
                if isinstance(node,ast.ImportFrom):
                    prefix=package[:len(package)-node.level+1] if node.level else []
                    module=prefix+(node.module.split('.') if node.module else [])
                elif isinstance(node,ast.Import):
                    for alias in node.names:
                        module=alias.name.split('.')
                        if module[0] in ('pipeline','training'):
                            candidate=ROOT.joinpath(*module)
                            if not (candidate.with_suffix('.py').is_file() or (candidate/'__init__.py').is_file()):
                                missing.append(('.'.join(parts),'.'.join(module)))
                    continue
                else:continue
                if module and module[0] in ('pipeline','training'):
                    candidate=ROOT.joinpath(*module)
                    if not (candidate.with_suffix('.py').is_file() or (candidate/'__init__.py').is_file()):
                        missing.append(('.'.join(parts),'.'.join(module)))
    assert not missing,missing
