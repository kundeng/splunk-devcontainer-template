#!/usr/bin/env node

/**
 * Adapt a @splunk/create-scaffolded app package for UCC overlay use.
 *
 * Applies the 6 mechanical transformations needed to turn a standalone
 * React Splunk app into one whose stage/ output merges into a UCC app.
 *
 * Usage:
 *   node bin/adapt-for-ucc.mjs --app-pkg <react-app-dir> \
 *                               --ucc-app <ucc_app_name> \
 *                               --component-pkg <component-package-name> \
 *                               --page <page-name>
 *
 * Example:
 *   node bin/adapt-for-ucc.mjs --app-pkg urc-app \
 *                               --ucc-app urc_app \
 *                               --component-pkg urc-builder \
 *                               --page builder
 */

import fs from 'node:fs';
import path from 'node:path';
import { parseArgs } from 'node:util';

const { values } = parseArgs({
    options: {
        'app-pkg':       { type: 'string' },
        'ucc-app':       { type: 'string' },
        'component-pkg': { type: 'string' },
        'page':          { type: 'string' },
    },
    strict: true,
});

const appPkg       = values['app-pkg'];
const uccApp       = values['ucc-app'];
const componentPkg = values['component-pkg'];
const pageName     = values['page'];

if (!appPkg || !uccApp || !componentPkg || !pageName) {
    console.error(
        'All options required: --app-pkg, --ucc-app, --component-pkg, --page\n' +
        'Example: node bin/adapt-for-ucc.mjs --app-pkg urc-app --ucc-app urc_app --component-pkg urc-builder --page builder'
    );
    process.exit(1);
}

const pkgDir = path.resolve('packages', appPkg);
if (!fs.existsSync(pkgDir)) {
    console.error(`Package directory not found: ${pkgDir}`);
    process.exit(1);
}

console.log(`Adapting ${appPkg} for UCC overlay (${uccApp})...`);

// ── Helper ──────────────────────────────────────────────────────────

function readFile(relPath) {
    return fs.readFileSync(path.join(pkgDir, relPath), 'utf8');
}

function writeFile(relPath, content) {
    const abs = path.join(pkgDir, relPath);
    fs.mkdirSync(path.dirname(abs), { recursive: true });
    fs.writeFileSync(abs, content, 'utf8');
    console.log(`  WRITE: ${relPath}`);
}

function renameFile(from, to) {
    const absFrom = path.join(pkgDir, from);
    const absTo   = path.join(pkgDir, to);
    if (fs.existsSync(absFrom)) {
        fs.mkdirSync(path.dirname(absTo), { recursive: true });
        fs.renameSync(absFrom, absTo);
        console.log(`  RENAME: ${from} → ${to}`);
    }
}

// ── 1. webpack.config.js ────────────────────────────────────────────
//    component.config → base.config + CopyWebpackPlugin + page discovery

writeFile('webpack.config.js', `const fs = require('fs');
const path = require('path');
const CopyWebpackPlugin = require('copy-webpack-plugin');
const { merge: webpackMerge } = require('webpack-merge');
const baseConfig = require('@splunk/webpack-configs/base.config').default;

// Discover page entry points dynamically from src/main/webapp/pages/
const entries = fs
    .readdirSync(path.join(__dirname, 'src/main/webapp/pages'))
    .filter((pageFile) => !/^\\./.test(pageFile))
    .reduce((accum, page) => {
        accum[page] = path.join(__dirname, 'src/main/webapp/pages', page);
        return accum;
    }, {});

module.exports = webpackMerge(baseConfig, {
    entry: entries,
    output: {
        path: path.join(__dirname, 'stage/appserver/static/pages/'),
        filename: '[name].js',
    },
    plugins: [
        new CopyWebpackPlugin({
            patterns: [
                {
                    from: path.join(__dirname, 'src/main/resources/splunk'),
                    to: path.join(__dirname, 'stage'),
                },
            ],
        }),
    ],
    devtool: 'eval-source-map',
});
`);

// ── 2. package.json ─────────────────────────────────────────────────
//    Add UCC-overlay deps, drop types:build from build script

const pkg = JSON.parse(readFile('package.json'));

pkg.dependencies = pkg.dependencies || {};
pkg.dependencies['@splunk/react-page'] = pkg.dependencies['@splunk/react-page'] || '^8.2.1';
pkg.dependencies[`@splunk/${componentPkg}`] = pkg.dependencies[`@splunk/${componentPkg}`] || '^0.0.1';

pkg.devDependencies = pkg.devDependencies || {};
pkg.devDependencies['copy-webpack-plugin'] = pkg.devDependencies['copy-webpack-plugin'] || '^11.0.0';

// Remove types:build from build script — wrapper doesn't export types
if (pkg.scripts?.build?.includes('types:build')) {
    pkg.scripts.build = pkg.scripts.build.replace(/\s*&&\s*yarn types:build/, '');
}

writeFile('package.json', JSON.stringify(pkg, null, 2) + '\n');

// ── 3. Rename page + rewrite entry point ────────────────────────────
//    Scaffold creates pages/<AppName>/, we want pages/<pageName>/

// Find the scaffolded page directory (first dir in pages/)
const pagesDir = path.join(pkgDir, 'src/main/webapp/pages');
const scaffoldedPages = fs.readdirSync(pagesDir).filter(d =>
    fs.statSync(path.join(pagesDir, d)).isDirectory() && !d.startsWith('.')
);

if (scaffoldedPages.length > 0 && scaffoldedPages[0] !== pageName) {
    renameFile(
        `src/main/webapp/pages/${scaffoldedPages[0]}`,
        `src/main/webapp/pages/${pageName}`
    );
}

// Derive component name from package name (kebab → PascalCase)
const componentName = componentPkg
    .replace(/(^|-)(\w)/g, (_, _2, c) => c.toUpperCase());

writeFile(`src/main/webapp/pages/${pageName}/index.tsx`, `import React from 'react';
import layout from '@splunk/react-page';
import { getUserTheme } from '@splunk/splunk-utils/themes';
import ${componentName} from '@splunk/${componentPkg}';

import { StyledContainer } from './Styles';

getUserTheme()
    .then((theme) => {
        layout(
            <StyledContainer>
                <${componentName} />
            </StyledContainer>,
            {
                theme,
            }
        );
    })
    .catch((e) => {
        const errorEl = document.createElement('span');
        errorEl.innerHTML = e;
        document.body.appendChild(errorEl);
    });
`);

// ── 4. nav/default.xml ──────────────────────────────────────────────
//    Add configuration + search views for UCC coexistence

writeFile('src/main/resources/splunk/default/data/ui/nav/default.xml',
`<nav>
    <view name="${pageName}" default="true"/>
    <view name="configuration"/>
    <view name="search"/>
</nav>
`);

// ── 5. views/<page>.xml ─────────────────────────────────────────────
//    Template path must reference UCC app name, not React package name

// Remove scaffold-generated view XML if different name
for (const old of scaffoldedPages) {
    const oldView = `src/main/resources/splunk/default/data/ui/views/${old}.xml`;
    if (fs.existsSync(path.join(pkgDir, oldView)) && old !== pageName) {
        fs.unlinkSync(path.join(pkgDir, oldView));
        console.log(`  DELETE: ${oldView}`);
    }
}

writeFile(`src/main/resources/splunk/default/data/ui/views/${pageName}.xml`,
`<?xml version="1.0"?>
<view template="${uccApp}:/templates/${pageName}.html" type="html">
\t<label>${pageName.charAt(0).toUpperCase() + pageName.slice(1).replace(/([A-Z])/g, ' $1').trim()}</label>
</view>
`);

// ── 6. templates/<page>.html ────────────────────────────────────────
//    page_path must reference UCC app name

// Remove scaffold-generated template if different name
for (const old of scaffoldedPages) {
    const oldTpl = `src/main/resources/splunk/appserver/templates/${old}.html`;
    if (fs.existsSync(path.join(pkgDir, oldTpl)) && old !== pageName) {
        fs.unlinkSync(path.join(pkgDir, oldTpl));
        console.log(`  DELETE: ${oldTpl}`);
    }
}

writeFile(`src/main/resources/splunk/appserver/templates/${pageName}.html`,
`<!DOCTYPE html>
<html class="no-js" lang="">
    <head>
        <meta charset="utf-8" />
        <meta http-equiv="x-ua-compatible" content="ie=edge" />
        <title>${uccApp.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
    </head>

    <body>
        <script src="\${make_url('/config?autoload=1')}" crossorigin="use-credentials"></script>
        <script src="\${make_url('/static/js/i18n.js')}"></script>
        <script src="\${make_url('/i18ncatalog?autoload=1')}"></script>
        <script>
            __splunkd_partials__ = \${json_decode(splunkd)};
        </script>

        <% page_path = "/static/app/${uccApp}/pages/" + page + ".js" %>

        <script src="\${make_url(page_path)}"></script>
    </body>
</html>
`);

console.log('Done. UCC overlay adaptations applied.');
console.log(`Next: cd react && yarn install && yarn workspace @splunk/${appPkg} build`);
