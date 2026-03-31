#!/usr/bin/env node

/**
 * Non-interactive wrapper for @splunk/create.
 *
 * Calls the @splunk/create generators programmatically with all options
 * pre-set, bypassing inquirer prompts. Requires @splunk/create as a
 * devDependency in the monorepo root (which pulls in mem-fs, etc.).
 *
 * Usage:
 *   node bin/splunk-create-noninteractive.mjs app       <AppName>                    # Splunk app + page + component
 *   node bin/splunk-create-noninteractive.mjs page      <PageName>  --app <pkg-dir>  # Add page + component to existing app
 *   node bin/splunk-create-noninteractive.mjs component <Name>      [--type basic|dashboard]  # Component library only
 *
 * Options:
 *   --force   Overwrite existing files (default: skip with warning)
 *   --type    Component type: "basic" (default) or "dashboard"
 *   --app     Package directory name of the target Splunk app (required for "page")
 */

import fs from 'node:fs';
import path from 'node:path';
import { parseArgs } from 'node:util';

import { create as createMemFs } from 'mem-fs';
import { create as createEditor } from 'mem-fs-editor';
import { validateName } from '@splunk/create/lib/utils/utils.js';

// ── Parse CLI ───────────────────────────────────────────────────────

const { values, positionals } = parseArgs({
    options: {
        app:   { type: 'string' },
        type:  { type: 'string', default: 'basic' },
        force: { type: 'boolean', default: false },
    },
    strict: false,
    allowPositionals: true,
});

const [command, name] = positionals;

if (!command || !name) {
    console.error(
        'Usage:\n' +
        '  node bin/splunk-create-noninteractive.mjs app       <AppName>\n' +
        '  node bin/splunk-create-noninteractive.mjs page      <PageName>  --app <pkg-dir>\n' +
        '  node bin/splunk-create-noninteractive.mjs component <Name>      [--type basic|dashboard]'
    );
    process.exit(1);
}

const validation = validateName(name);
if (validation !== true) {
    console.error(`Invalid name "${name}": ${validation}`);
    process.exit(1);
}

// ── Run generators ──────────────────────────────────────────────────

const store = createMemFs();
const fsStore = createEditor(store);

console.log(`@splunk/create (non-interactive): ${command} "${name}"...`);

if (command === 'app') {
    const { default: ReactSplunkApp }  = await import('@splunk/create/lib/ReactSplunkAppGenerator.js');
    const { default: ReactSplunkPage } = await import('@splunk/create/lib/ReactSplunkPageGenerator.js');
    const { default: ReactComponent }  = await import('@splunk/create/lib/ReactComponentGenerator.js');

    const appInfo  = await ReactSplunkApp(fsStore, { appName: name });
    const pageName = await ReactSplunkPage(fsStore, { existingApp: appInfo.packageName, pageName: name });
    await ReactComponent(fsStore, { componentName: pageName, componentType: values.type });

} else if (command === 'page') {
    if (!values.app) {
        console.error('--app <pkg-dir> is required for the "page" command');
        process.exit(1);
    }
    const { default: ReactSplunkPage } = await import('@splunk/create/lib/ReactSplunkPageGenerator.js');
    const { default: ReactComponent }  = await import('@splunk/create/lib/ReactComponentGenerator.js');

    const pageName = await ReactSplunkPage(fsStore, { existingApp: values.app, pageName: name });
    await ReactComponent(fsStore, { componentName: pageName, componentType: values.type });

} else if (command === 'component') {
    const { default: ReactComponent } = await import('@splunk/create/lib/ReactComponentGenerator.js');
    await ReactComponent(fsStore, { componentName: name, componentType: values.type });

} else {
    console.error(`Unknown command "${command}". Use: app, page, component`);
    process.exit(1);
}

// ── Write files (skip existing unless --force) ──────────────────────

const memPaths = Object.keys(fsStore.dump());

for (const p of memPaths) {
    const abs = path.join(process.cwd(), p);
    if (fs.existsSync(abs) && !values.force) {
        // Restore original content so commit is a no-op for this file
        fsStore.write(abs, fs.readFileSync(abs));
        console.log(`  SKIP: ${p}  (use --force to overwrite)`);
    } else {
        console.log(`  ${fs.existsSync(abs) ? 'OVERWRITE' : 'CREATE'}: ${p}`);
    }
}

await fsStore.commit();
console.log('Done.');
