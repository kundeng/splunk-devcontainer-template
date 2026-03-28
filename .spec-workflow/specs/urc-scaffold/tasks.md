# URC Scaffold — Tasks

- [x] 1. Scaffold UCC app via ucc-gen init
  - Run `task ucc:init APP_NAME=urc_app` to create the skeleton
  - Verify globalConfig.json, package/, app.manifest are created
  - Run `task ucc:build APP_NAME=urc_app` to confirm baseline build works
  - Purpose: Establish working UCC project structure
  - _Leverage: Existing `task ucc:init` and `task ucc:build` primitives_
  - _Requirements: R1_
  - _Prompt: Implement the task for spec urc-scaffold, first run spec-workflow-guide to get the workflow guide then implement the task. Role: Splunk Add-on Developer | Task: Scaffold the URC app using existing Taskfile primitives. Run `task ucc:init APP_NAME=urc_app` then `task ucc:build APP_NAME=urc_app`. Verify the output structure is correct. | Restrictions: Use existing task primitives only. Do not modify Taskfile.yml unless a task is broken. Do not add any connector logic. | _Leverage: Taskfile.yml ucc:init and ucc:build tasks | _Requirements: R1 | Success: `ucc/urc_app/` exists with globalConfig.json and package/. `ucc/output/urc_app/` is produced by ucc:build without errors. Mark task [-] when starting and [x] when complete in tasks.md, and log implementation with log-implementation tool._

- [x] 2. Download Airbyte declarative component schema
  - Fetch `declarative_component_schema.yaml` from airbytehq/airbyte-python-cdk main branch
  - Store at `ucc/urc_app/schema/declarative_component_schema.yaml`
  - Verify it is valid YAML and contains the expected `definitions:` section
  - Purpose: Have the canonical schema available in-repo for code generation and reference
  - _Leverage: curl or wget, raw GitHub URL_
  - _Requirements: R2_
  - _Prompt: Implement the task for spec urc-scaffold, first run spec-workflow-guide to get the workflow guide then implement the task. Role: DevOps Engineer | Task: Download the Airbyte declarative component schema YAML from https://raw.githubusercontent.com/airbytehq/airbyte-python-cdk/main/airbyte_cdk/sources/declarative/declarative_component_schema.yaml and save it to `ucc/urc_app/schema/declarative_component_schema.yaml`. Verify the file is valid YAML. | Restrictions: Do not modify the schema content. Just download and store. | _Leverage: curl, python3 -c "import yaml" for validation | _Requirements: R2 | Success: Schema file exists at the specified path, is valid YAML, and contains 100+ definitions. Mark task [-] when starting and [x] when complete in tasks.md, and log implementation with log-implementation tool._

- [x] 3. Full CDK dependency audit
  - Run `pip install airbyte-cdk --target /tmp/airbyte-cdk-audit` in a temp dir
  - Measure total size
  - List all packages, categorize as pure-Python vs C-extension (find *.so files)
  - Identify which C-extension packages are in the declarative runtime hot path
  - Document findings in `ucc/urc_app/AUDIT.md`
  - Purpose: Get real data on what the CDK brings in, instead of guessing
  - _Requirements: R3, R5_
  - _Prompt: Implement the task for spec urc-scaffold, first run spec-workflow-guide to get the workflow guide then implement the task. Role: Python Packaging Specialist | Task: Install airbyte-cdk into a temp directory via `pip install airbyte-cdk --target /tmp/airbyte-cdk-audit`. Then audit the results - measure total size with `du -sh`, find all .so files with `find . -name "*.so"`, list all top-level package directories. Categorize each package as pure-Python or C-extension. Write the findings to `ucc/urc_app/AUDIT.md`. | Restrictions: Do NOT install into the actual app lib/ directory. This is an audit only in a temp directory. | _Requirements: R3, R5 | Success: AUDIT.md exists with total size, package list, C-extension vs pure-Python categorization, and analysis of which C extensions are avoidable. Mark task [-] when starting and [x] when complete in tasks.md, and log implementation with log-implementation tool._

- [x] 4. Create minimal requirements.txt for pure-Python subset
  - Based on audit results, create `package/lib/requirements.txt` with only the pure-Python deps needed for the declarative runtime
  - Create `package/lib/exclude.txt` to strip unwanted transitive deps
  - Test install with `pip install --target /tmp/urc-deps-test -r requirements.txt` and verify no .so files
  - Purpose: Define the minimal, pure-Python dependency set that will actually ship
  - _Leverage: AUDIT.md findings from task 3_
  - _Requirements: R3_
  - _Prompt: Implement the task for spec urc-scaffold, first run spec-workflow-guide to get the workflow guide then implement the task. Role: Python Packaging Specialist | Task: Using the AUDIT.md findings, create `ucc/urc_app/package/lib/requirements.txt` with the minimal pure-Python dependencies needed for the Airbyte CDK declarative runtime. Also create `ucc/urc_app/package/lib/exclude.txt` to strip C-extension packages and unnecessary transitive deps. Test the install into a temp dir and verify zero .so files. | Restrictions: No C extensions allowed. If a package has an optional C speedup (like MarkupSafe), include it but verify the pure-Python fallback works. Pin versions for reproducibility. | _Requirements: R3 | Success: requirements.txt and exclude.txt exist. `pip install --target /tmp/test -r requirements.txt` produces zero .so files. Total size is documented. Mark task [-] when starting and [x] when complete in tasks.md, and log implementation with log-implementation tool._

- [x] 5. Build UCC app with vendored deps
  - Run `task ucc:build APP_NAME=urc_app` which will install requirements.txt into lib/
  - Verify build completes without errors
  - Check output lib/ directory for expected packages
  - Measure final app size
  - Purpose: Verify the UCC build pipeline works with our dependency set
  - _Leverage: task ucc:build, requirements.txt from task 4_
  - _Requirements: R4_
  - _Prompt: Implement the task for spec urc-scaffold, first run spec-workflow-guide to get the workflow guide then implement the task. Role: Splunk Add-on Developer | Task: Run `task ucc:build APP_NAME=urc_app` and verify it completes successfully. Check `ucc/output/urc_app/lib/` contains the expected vendored packages. Measure the total output size. Update AUDIT.md with build results. | Restrictions: Do not modify the build pipeline. If it fails, document the error and investigate. | _Requirements: R4 | Success: ucc:build completes without errors. output/urc_app/lib/ contains pydantic, dpath, yaml, jinja2, requests etc. AUDIT.md updated with build results and output size. Mark task [-] when starting and [x] when complete in tasks.md, and log implementation with log-implementation tool._

- [x] 6. Load in Splunk 10.2 and verify imports
  - Link the built app into Splunk via `task ucc:link APP_NAME=urc_app`
  - Restart Splunk and verify the app appears in Splunk Web
  - Test key imports from Splunk's Python (via a test script or REST endpoint)
  - Document what works and what fails
  - Update AUDIT.md with final results and recommendations
  - Purpose: End-to-end verification that vendored deps load under Splunk's Python
  - _Leverage: task ucc:link, task dev:restartd_
  - _Requirements: R4, R5_
  - _Prompt: Implement the task for spec urc-scaffold, first run spec-workflow-guide to get the workflow guide then implement the task. Role: Splunk Add-on Developer | Task: Link the built URC app into the running Splunk 10.2 instance via `task ucc:link APP_NAME=urc_app` and `task dev:refresh` or `task dev:restartd`. Verify the app appears in Splunk Web at localhost:8000. Create a simple test script that attempts to import key packages (import_declare_test, pydantic, dpath, yaml, jinja2, requests) and run it under Splunk's Python. Document results in AUDIT.md. | Restrictions: The Splunk container must be running (`task dev:up` if needed). Do not modify Splunk's Python environment. | _Requirements: R4, R5 | Success: App appears in Splunk Web. Import test results documented in AUDIT.md showing which packages load successfully and which fail (with exact error messages). Final recommendations section added. Mark task [-] when starting and [x] when complete in tasks.md, and log implementation with log-implementation tool._
