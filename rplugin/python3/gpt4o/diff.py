import os
import pathlib
import tempfile
import subprocess
from typing import Optional

def compute_diff(self, new_content: str, old_content: str, current_path: str = '') -> Optional[str]:
    old_label = 'a'
    new_label = 'b'
    if current_path:
        current_path = '/' + current_path.lstrip('/')
        old_label += current_path
        new_label += current_path

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = pathlib.Path(temp_dir)
        old_file = temp_dir / 'a'
        new_file = temp_dir / 'b'
        with open(old_file, 'w') as f:
            f.write(old_content)
        with open(new_file, 'w') as f:
            f.write(new_content)
        with subprocess.Popen(['diff', '-u', '-d',
                               '--label', old_label,
                               '--label', new_label,
                               str(old_file), str(new_file),
                               ],
                              stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              env=dict(os.environ, LC_ALL='C'),
                              ) as proc:
            output, error = proc.communicate()
            if error:
                self.alert(error.decode('utf-8'), 'ERROR')
            ret = proc.wait()
            if ret == 0:
                return None
            if ret != 1:
                self.alert(f'diff exited with {ret}', 'ERROR')
                return None

    diff = output.decode('utf-8')
    return diff

