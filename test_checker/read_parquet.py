import pyarrow.parquet as pq
import sys
import glob

def construct_usi(row) -> None:
    try:
        usi = f'mzspec:{row["project_accession"]}:{row["ms_run"]}:{row["index_type"]}:{row["index_number"]}'
        print(usi)
    except Exception as e:
        print("error:", e, file=sys.stderr)

# for location in ["/sc/projects/sci-renard/usi-grabber/shared/mgf_files/final/with_phospho/", "/sc/projects/sci-renard/usi-grabber/shared/mgf_files/final/without_phospho/"]:
for location in ["/sc/projects/sci-renard/usi-grabber/shared/mgf_files/final/without_phospho/"]:
    for file in glob.glob(pathname=f"{location}**/*.parquet", recursive=True):
        print(file, file=sys.stderr)

        table = pq.read_table(file)

        # Convert the Arrow Table to a Pandas DataFrame
        df = table.to_pandas()


       

        df.apply(construct_usi, axis=1)
