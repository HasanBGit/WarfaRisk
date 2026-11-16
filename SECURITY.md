# Security policy

WarfaRisk is a research prototype and is not a clinical tool. Do not use it for dosing decisions.

## Reporting a vulnerability

Please report security problems privately through GitHub: open the Security tab of this repository and choose "Report a vulnerability". Do not open a public issue for a vulnerability.

Do not include patient data or credentials in a report.

## Supported version

Only the version tagged `v1.0-saip` and the current `main` branch are supported.

## Model files

Trained AutoGluon models are stored as pickle files, which can run code when loaded. Load them only from the official Hugging Face repositories linked in the README.
