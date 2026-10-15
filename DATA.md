# AIDev input data and integrity

**Original dataset:** AIDev: Studying AI Coding Agents on GitHub (Li, Zhang and Hassan).

- Public project: https://github.com/SAILResearch/AI_Teammates_in_SE3
- Public dataset catalog: https://huggingface.co/datasets/hao-li/AIDev
- Versioned public archival record: https://zenodo.org/records/16919272

The source corpus changes between releases. Exact reproduction requires the **same
four input tables** that produced the archived digests below; simply downloading
the latest version may not reproduce the cohort or splits. The tables are **not
redistributed** in this repository. The original AIDev files may need to be selected
and renamed to the filenames expected by the frozen drivers.

Store these files locally under `D:\IJIES_STOP_APR\data\`:

| Filename | Frozen SHA-256 |
| --- | --- |
| `aidev_pull_request.parquet` | `08f520b5ef36b6281f82090db09ac14aefc3534ee113886e7bc85131fd8d214c` |
| `aidev_pr_reviews.parquet` | `94051ff602c0ac560dbbc6b34826c1ec1353277d33744718443bc9f3b0532afa` |
| `aidev_pr_comments.parquet` | `fa208bafadd5196c6ff414270da4d33a9b13fb65fea47207920ee6df070bf316` |
| `aidev_pr_timeline.parquet` | `9f45c52c0a7ffa4395ea6c7ce5beb993b3dfb2425ef3848c6d28e2e96471a6f8` |

To check a local file with PowerShell:

```powershell
Get-FileHash "D:\IJIES_STOP_APR\data\aidev_pull_request.parquet" -Algorithm SHA256
```

The driver checks source/data fingerprints before evaluation. Consult
`outputs/08_final_evidence_summary.json` for the full integrity record.

No source Parquet records or local credentials are included in the repository.
