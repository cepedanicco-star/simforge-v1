"""Keep Windows DLL lookup process-local for an existing native Python runtime."""
import os
from pathlib import Path
import sys

_handles=[]


def import_torch():
    if os.name=='nt' and not _handles:
        root=Path(sys.executable).resolve().parent
        for folder in (root/'Library/bin',root/'Lib/site-packages/torch/lib'):
            if folder.is_dir():
                os.environ['PATH']=str(folder)+os.pathsep+os.environ.get('PATH','')
                _handles.append(os.add_dll_directory(str(folder)))
    import torch
    return torch
