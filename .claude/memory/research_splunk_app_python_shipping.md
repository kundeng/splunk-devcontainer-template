# Research: Shipping Python / Binary Deps with Splunk Apps

## Question
Can a Splunk app ship its own Python interpreter, venv, or large binary dependencies (numpy, pandas, pydantic-core, etc.)?

## Key Finding: YES -- Splunk's Own PSC App Is the Precedent

**Python for Scientific Computing (PSC)** is a Splunk-published app on Splunkbase that ships:
- A **complete Miniconda-based Python virtual environment**
- Binary dependencies: numpy, scipy, pandas, scikit-learn, statsmodels, cryptography, pydantic
- ~388 MB download size (Linux 64-bit, v4.3.1)
- Python 3.13 as of v4.3.0
- **Both Cloud and Enterprise compatible**
- Apache 2.0 license
- Splunkbase IDs: 2881 (Mac), 2882 (Linux), 2883 (Windows)

This is THE reference for shipping binary deps with a Splunk app.

## AppInspect Rules for Binary Files

### Check: `check_for_bin_files`
- **Tags**: `splunk_appinspect`, `appapproval`, `cloud`
- Fails if `.exe` files found outside `bin/`
- Fails if files outside `bin/` have execute permissions
- `bin/` directory is **excluded** from this check

### Check: `check_for_executable_flag`
- **Tags**: `splunk_appinspect`, `appapproval`, `manual`, `cloud`
- Uses `file` command magic to detect executables
- Skips `bin/` and `appserver/controllers/`
- Produces **manual_check** (not failure) -- triggers human review

### Check: `check_for_binary_files_without_source_code`
- **Tags**: `splunk_appinspect`, `manual`, `cloud`
- Requires binary files have matching source code OR...
- **Binary File Declaration**: Include `# Binary File Declaration` section in README.txt
- This causes the check to **omit** the declared binary files from failing

### No checks explicitly prohibit:
- Shipping a Python interpreter
- Shipping a venv/conda environment
- Shipping .so shared libraries
- Having large packages

## Splunkbase Package Size

- **Files > 200 MB**: require documented justification
- **Splunk Web upload limit**: 512 MB default (configurable via `max_upload_size` in web.conf)
- PSC at 388 MB passes both AppInspect and Splunkbase
- No hard maximum found for Splunkbase submission itself

## AppInspect Tags: Cloud vs Enterprise

| Tag | Purpose |
|-----|---------|
| `splunk_appinspect` | Attached to every check |
| `appapproval` | Splunkbase submission |
| `cloud` | Cloud vetting (stricter) |
| `manual` | Triggers human review |
| `private_victoria` | Splunk Cloud Victoria self-service |
| `private_classic` | Splunk Cloud Classic self-service |

**Key insight**: For enterprise-only apps, you only need to pass `appapproval` tag checks. The `cloud` tag adds stricter requirements for cloud deployments. Architecture-specific tags "have no impact to customer managed Splunk Enterprise deployments."

## Subprocess Approach (Alternative)

A modular input can use subprocess to call an external Python:
```python
subprocess.Popen(['/path/to/venv/bin/python', 'script.py'], ...)
```
- Must clear `PYTHONPATH` to avoid Splunk's Python contaminating the venv
- Wrapper shell scripts can activate/deactivate venv
- Works for Enterprise; NOT compatible with Cloud

## What's Prohibited

- `.pyc` / `.pyo` files -- Splunkbase explicitly bans compiled Python bytecode
- Hidden files (dotfiles)
- Plain text credentials
- Invasive relative paths (`../`)
- `thumbs.db`

## Practical Implications for URC App

1. **Shipping a vendored venv with airbyte-cdk + binary deps IS viable** -- PSC does exactly this
2. **README must include `# Binary File Declaration`** listing all .so files
3. **Package will be large** (~200-400 MB) but within Splunkbase norms
4. **Enterprise-only is easier** -- skip cloud tag, fewer restrictions
5. **Platform-specific packages** may be needed (like PSC's Linux/Mac/Windows split)
6. **Build process**: Use conda/pip to create a portable env, bundle in app's `bin/` dir

## Sources
- PSC GitHub: https://github.com/splunk/Splunk-python-for-scientific-computing
- PSC Splunkbase: https://splunkbase.splunk.com/app/2882
- AppInspect checks source: https://github.com/splunkdevabhi/appinspect
- Splunkbase file standards: https://dev.splunk.com/enterprise/docs/releaseapps/splunkbase/approvalcriteria
- AppInspect tag reference: https://dev.splunk.com/enterprise/reference/appinspect/appinspecttagreference
- Self-service tags blog: https://www.splunk.com/en_us/blog/platform/greater-self-service-private-apps-on-cloud-with-new-appinspect-tags.html
