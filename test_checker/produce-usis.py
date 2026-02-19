from pyteomics import mgf
from pathlib import Path
import sys

for file_path in ["/sc/projects/sci-renard/usi-grabber/shared/mgf_files/validation_sp_p.mgf", "/sc/projects/sci-renard/usi-grabber/shared/mgf_files/validation_sp_non_p.mgf"]:
    print(file_path, file=sys.stderr)
    def construct_usi(entry) -> str:
        params = entry["params"]
        file_stem = Path(params["provenance_filename"]).stem

        return f'mzspec:{params["provenance_dataset_pxd"]}:{file_stem}:scan:{params["provenance_scan"]}'
    with mgf.read(file_path, use_index=False) as reader:
        for entry in reader:
            try:
                print(construct_usi(entry))
            except Exception as e:
                print(e, file=sys.stderr)