# The Splunk UCC React Tooling Stack: A Ground-Up Explanation

> **Goal:** You want to build a custom React UI for a Splunk add-on. You open the project
> and find 14 config files per package, three webpack configs, two compilers (Babel AND
> TypeScript?), XML views, Mako templates, and a Taskfile orchestrating all of it.
>
> This doc explains every piece — what it does, why it exists, and how it connects to
> the others. It starts from the individual tools (the leaves) and builds up to the
> full pipeline (the tree). Every code snippet is from the actual files in
> `/workspace/react/`, not hypothetical examples.
>
> **Audience:** Developers who can write React components but haven't built a Splunk
> app before, or anyone inheriting this project who needs to understand the build
> system before making changes.

---

## Table of Contents

1. [The Big Picture: What Problem Are We Solving?](#1-the-big-picture)
2. [The Cast of Characters (Tools)](#2-the-cast-of-characters)
3. [Yarn Workspaces: The Monorepo Glue](#3-yarn-workspaces)
4. [Babel: The Translator](#4-babel)
5. [TypeScript: The Type Checker (That Doesn't Compile)](#5-typescript)
6. [Webpack: The Bundler (The Big One)](#6-webpack)
7. [How Splunk Actually Loads Your React Page](#7-how-splunk-loads-your-react-page)
8. [The Three Webpack Configs Per Package](#8-three-webpack-configs)
9. [Component vs Application: Two Different Build Strategies](#9-component-vs-application)
10. [The Full Build Flow: What Happens When You Run `yarn build`](#10-full-build-flow)
11. [ESLint, Prettier, Stylelint: The Quality Tools](#11-quality-tools)
12. [Jest: The Test Runner](#12-jest)
13. [Dependencies vs DevDependencies vs PeerDependencies](#13-dependency-types)
14. [The Scaffolding Pipeline: From Nothing to Running App](#14-the-scaffolding-pipeline)
15. [The UCC Overlay: How React Merges Into a UCC App](#15-the-ucc-overlay)
16. [The Taskfile: Orchestrating Everything](#16-the-taskfile)
17. [Mental Model: The Complete Pipeline Diagram](#17-complete-diagram)

---

## 1. The Big Picture

You write files like `UrcBuilder.tsx` — modern TypeScript with JSX, ES modules, `import` statements.
Splunk's web framework expects a single `.js` file per page, using CommonJS (`require()`), that runs in the browser.

**The entire tooling stack exists to bridge that gap.**

```
What you write:                    What Splunk needs:
─────────────────                  ──────────────────
UrcBuilder.tsx (TypeScript+JSX)    builder.js (plain JS, single file)
import React from 'react'    -->  All dependencies bundled or externalized
import Button from '@splunk/...'  Works in browsers (no import/export)
Spread across many files           One file per page
```

Every tool in the stack handles one piece of this transformation:

| Tool | Job |
|------|-----|
| **Babel** | Translates modern syntax (JSX, TS, ES2024) down to browser-compatible JS |
| **TypeScript** | Checks types, generates `.d.ts` files (does NOT compile in this setup) |
| **Webpack** | Bundles many files into few, resolves `import`s, applies loaders |
| **Yarn Workspaces** | Manages the monorepo so packages can depend on each other |

---

## 2. The Cast of Characters

Here's every config file in your project and what it controls:

```
react/                               <-- monorepo root
├── package.json                     <-- Yarn workspace definition + root scripts
├── babel.config.js                  <-- Tells Babel "look for .babelrc in packages/*"
├── yarn.lock                        <-- Exact dependency versions (auto-generated)
├── .prettierrc                      <-- Code formatting rules
├── .editorconfig                    <-- Editor-level formatting (tabs, newlines)
├── bin/
│   ├── splunk-create-noninteractive.mjs  <-- Automated scaffolding (bypasses prompts)
│   └── adapt-for-ucc.mjs                <-- 6 transformations: standalone → UCC overlay
│
└── packages/
    ├── urc-builder/                 <-- COMPONENT LIBRARY (your actual UI)
    │   ├── package.json             <-- This package's deps, scripts, metadata
    │   ├── .babelrc.js              <-- Babel preset for this package
    │   ├── tsconfig.json            <-- TypeScript options
    │   ├── webpack.config.js        <-- Production build config (component mode)
    │   ├── .eslintrc.js             <-- Linting rules
    │   ├── stylelint.config.js      <-- CSS-in-JS linting
    │   ├── jest.config.js           <-- Test runner config
    │   ├── bin/build.js             <-- Shell script wrapper for webpack CLI
    │   ├── src/                     <-- Your source code
    │   │   ├── index.ts             <-- Package entry point
    │   │   └── UrcBuilder.tsx       <-- The actual component
    │   └── demo/
    │       ├── demo.tsx                      <-- Demo app entry point
    │       ├── webpack.standalone.config.js  <-- Build for localhost:8080
    │       └── webpack.splunkapp.config.js   <-- Build for Splunk dev
    │
    └── urc-app/                     <-- SPLUNK APP WRAPPER (mounts component into Splunk)
        ├── webpack.config.js        <-- Discovers pages, bundles everything, copies resources
        ├── src/
        │   ├── main/webapp/pages/   <-- Webpack entry points (one per Splunk page)
        │   │   └── builder/
        │   │       └── index.tsx    <-- THE entry point Splunk loads
        │   └── main/resources/splunk/  <-- XML views, HTML templates, app.conf
        └── stage/                   <-- BUILD OUTPUT → merges into UCC app
```

**Key insight:** Most of these files are *configuration*, not code. The actual application logic lives only in `src/`. Everything else tells tools how to process that source.

---

## 3. Yarn Workspaces: The Monorepo Glue

Your root `package.json` declares a **workspace**:

```json
// react/package.json
{
  "private": true,
  "workspaces": ["packages/*"],
  "scripts": {
    "build": "yarn workspaces run build",
    "start": "yarn workspace @splunk/urc-app run start"
  }
}
```

### What workspaces do:

```
WITHOUT workspaces:                  WITH workspaces:
──────────────────                   ─────────────────
packages/urc-app/node_modules/       react/node_modules/        <-- shared!
  ├── react                            ├── react                <-- installed once
  ├── webpack                          ├── webpack
  └── ...                              └── @splunk/
packages/urc-builder/node_modules/        └── urc-builder -> ../../packages/urc-builder
  ├── react                                                    ↑ SYMLINK!
  ├── webpack
  └── ...
```

**Three things workspaces give you:**

1. **Hoisting**: Shared dependencies install once at the root `node_modules/`, not in each package. Saves disk space and ensures everyone uses the same version of React.

2. **Symlinking**: When `urc-app` depends on `@splunk/urc-builder`, Yarn creates a symlink from `node_modules/@splunk/urc-builder` to `packages/urc-builder/`. No need to publish to npm first.

3. **Cross-package commands**: `yarn workspaces run build` runs the `build` script in every package. `yarn workspace @splunk/urc-app run start` runs `start` in just one.

### How packages reference each other:

```json
// packages/urc-app/package.json
{
  "dependencies": {
    "@splunk/urc-builder": "^0.0.1"  // <-- resolved via symlink, not npm
  }
}
```

Then in code:
```tsx
// packages/urc-app/src/main/webapp/pages/builder/index.tsx
import UrcBuilder from '@splunk/urc-builder';  // follows the symlink
```

This import resolves to `packages/urc-builder/src/index.ts` (because that package's `"main"` field points to `"src/index.ts"`).

---

## 4. Babel: The Translator

Babel transforms syntax. It turns things browsers can't understand into things they can.

### The two-level config:

```js
// react/babel.config.js (ROOT — project-wide)
module.exports = {
    babelrcRoots: ['./packages/*'],  // "allow each package to have its own .babelrc"
};
```

```js
// packages/urc-builder/.babelrc.js (PACKAGE — actual transforms)
module.exports = {
    presets: ['@splunk/babel-preset'],
};
```

**Why two files?** Babel has a confusing config hierarchy:
- `babel.config.js` (root) = project-wide settings. It's the "boss".
- `.babelrc.js` (per-package) = package-specific transforms. But Babel ignores these by default for files outside the root. `babelrcRoots` tells Babel "these directories are allowed to have their own `.babelrc`".

### What `@splunk/babel-preset` does (conceptually):

```
INPUT (what you write):                    OUTPUT (what Babel produces):
───────────────────────                    ─────────────────────────────

// JSX                                     // Plain function calls
<Button label="Click" />            →      React.createElement(Button, { label: "Click" })

// TypeScript type annotations             // Types stripped entirely
const x: string = "hello"           →      const x = "hello"

// Optional chaining                       // Ternary fallback
user?.name                           →      user === null ? void 0 : user.name

// ES modules                              // CommonJS (for Splunk compat)
import React from 'react'           →      const React = require('react')
export default MyComponent           →      module.exports.default = MyComponent
```

Babel doesn't understand what your code *means*. It pattern-matches syntax and rewrites it. Each "preset" is a bundle of these transform rules.

### How Babel gets invoked:

Babel does NOT run on its own. **Webpack calls it** via `babel-loader`. From Splunk's `base.config.js`:

```js
// Inside @splunk/webpack-configs/base.config.js
{
    test: /\.[j|t]sx?$/,              // Match .js, .jsx, .ts, .tsx files
    exclude: /(node_modules)/,         // Don't transform dependencies
    use: [{
        loader: 'babel-loader',        // Webpack says: "run this file through Babel"
        options: {
            cacheDirectory: true,      // Cache results for speed
            rootMode: 'upward-optional' // Look upward for babel.config.js
        }
    }]
}
```

**The chain:** Webpack reads a `.tsx` file → hands it to `babel-loader` → Babel applies `@splunk/babel-preset` → returns plain JS → Webpack continues bundling.

---

## 5. TypeScript: The Type Checker (That Doesn't Compile)

This is the most confusing part for many people. **In your setup, TypeScript does NOT compile your code.**

```json
// packages/urc-builder/tsconfig.json
{
  "compilerOptions": {
    "emitDeclarationOnly": true,   // <-- KEY: only output .d.ts files
    "declaration": true,            // Generate type declarations
    "declarationDir": "./types",    // Put them in types/ folder
    "jsx": "react",                 // Understand JSX syntax
    "strict": true                  // Enable strict type checking
  },
  "include": ["src"]
}
```

### The division of labor:

```
                    Babel                          TypeScript (tsc)
                    ─────                          ────────────────
Input:              .tsx source files               .tsx source files (same!)
Output:             .js files (runnable code)       .d.ts files (type info only)
Does it type-check? No (strips types blindly)       Yes (reports errors)
Used by:            Webpack (via babel-loader)       Separate "types:build" script
When:               Every build                      Optionally, for publishing
```

```
UrcBuilder.tsx ──┬──→ [Babel] ──→ UrcBuilder.js    (runnable code, bundled by webpack)
                 │
                 └──→ [tsc]   ──→ types/UrcBuilder.d.ts  (just the type signatures)
```

**Why not use TypeScript for both?** Speed. Babel is faster because it doesn't type-check — it just strips types. The `@splunk/babel-preset` includes `@babel/preset-typescript` which knows how to remove `: string`, `interface Foo {}`, etc. without understanding them.

Your IDE (VS Code) uses `tsconfig.json` for real-time type checking as you edit. The `tsc` command is only run explicitly via `yarn types:build`.

---

## 6. Webpack: The Bundler (The Big One)

Webpack is the most complex tool here. Its job: take a tree of `import` statements and produce bundle files.

### Core concepts:

```
┌─────────────────────────────────────────────────────────┐
│                     WEBPACK                              │
│                                                          │
│  ENTRY ──→ MODULE GRAPH ──→ LOADERS ──→ PLUGINS ──→ OUTPUT │
│                                                          │
│  "Start    "Follow all      "Transform   "Post-    "Write│
│   here"     imports"         each file"   process"  files"│
└─────────────────────────────────────────────────────────┘
```

**Entry**: Where webpack starts reading. It follows every `import` from this file.
**Loaders**: Transform individual files (Babel for .tsx, css-loader for .css).
**Plugins**: Operate on the whole bundle (copy files, generate HTML, define globals).
**Output**: Where the final `.js` file(s) go.

### Your actual webpack config (urc-app):

```js
// packages/urc-app/webpack.config.js

const baseConfig = require('@splunk/webpack-configs/base.config').default;

// Step 1: Discover page entry points dynamically
const entries = fs
    .readdirSync(path.join(__dirname, 'src/main/webapp/pages'))
    .filter((pageFile) => !/^\./.test(pageFile))
    .reduce((accum, page) => {
        accum[page] = path.join(__dirname, 'src/main/webapp/pages', page);
        return accum;
    }, {});
// Result: { builder: '/...../pages/builder' }

module.exports = webpackMerge(baseConfig, {
    entry: entries,
    output: {
        path: path.join(__dirname, 'stage/appserver/static/pages/'),
        filename: '[name].js',  // [name] = "builder" → builder.js
    },
    plugins: [
        new CopyWebpackPlugin({
            patterns: [{
                from: 'src/main/resources/splunk',  // XML views, HTML templates
                to: 'stage',
            }],
        }),
    ],
});
```

### What `webpack-merge` does:

```js
webpackMerge(baseConfig, yourConfig)
```

It deep-merges two webpack configs. Splunk's `base.config` provides:
- Babel loader for .tsx/.jsx/.ts/.js files
- Font/image asset handling
- `resolve.extensions` so you can `import './Foo'` without `.tsx`
- Production/development mode switching via `NODE_ENV`
- `styled-components` attribute randomization

Your config adds:
- Entry points (which files to start bundling from)
- Output paths (where bundles go)
- Extra plugins (CopyWebpackPlugin)

The merge is *additive* — your rules add to the base rules, they don't replace them.

### What Splunk's base.config actually provides:

```js
// Simplified from @splunk/webpack-configs/base.config.js
{
    mode: DEBUG ? 'development' : 'production',

    module: {
        rules: [
            // Babel loader — THE transform pipeline
            {
                test: /\.[j|t]sx?$/,
                exclude: /node_modules/,
                use: ['babel-loader']
            },
            // Font files — inline small ones, emit large ones
            {
                test: /\.(woff|woff2|ttf)$/,
                type: 'asset',
                parser: { dataUrlCondition: { maxSize: 300000 } }
            },
            // Images — always emit as separate files
            {
                test: /\.(png|jpg|svg|gif)$/,
                type: 'asset/resource'
            },
            // Lodash AMD fix (Splunk-specific workaround)
            {
                include: /lodash/,
                use: ['imports-loader']
            }
        ]
    },

    resolve: {
        extensions: ['.wasm', '.mjs', '.js', '.jsx', '.ts', '.tsx']
        //            ↑ This is why you can write: import './Foo'
        //              and webpack finds Foo.tsx
    },

    plugins: [
        // Generates unique prefix for styled-components
        new webpack.DefinePlugin({
            'process.env.SC_ATTR': '"sc" + randomHex'
        })
    ]
}
```

---

## 7. How Splunk Actually Loads Your React Page

This is the **end-to-end journey** from a user clicking a Splunk nav link to your React component rendering. This is the part most people never fully understand:

### Step 1: Splunk XML View Definition

```xml
<!-- src/main/resources/splunk/default/data/ui/views/builder.xml -->
<view template="urc_app:/templates/builder.html" type="html">
    <label>Builder</label>
</view>
```

This tells Splunk: "when someone navigates to this view, render `builder.html` as a Mako template."

### Step 2: The HTML Template (Mako)

```html
<!-- src/main/resources/splunk/appserver/templates/builder.html -->
<body>
    <!-- Splunk injects its runtime config -->
    <script src="${make_url('/config?autoload=1')}"></script>
    <script src="${make_url('/static/js/i18n.js')}"></script>
    <script src="${make_url('/i18ncatalog?autoload=1')}"></script>
    <script>
        __splunkd_partials__ = ${json_decode(splunkd)};
    </script>

    <!-- Load YOUR webpack bundle -->
    <% page_path = "/static/app/urc_app/pages/" + page + ".js" %>
    <script src="${make_url(page_path)}"></script>
</body>
```

`${make_url(...)}` is Splunk's server-side Mako template syntax. It generates URLs with auth tokens and locale prefixes.

`__splunkd_partials__` is a global variable that gives your JS access to the Splunk REST API endpoint, current user, app context, etc.

The `page` variable comes from the URL: `/app/urc_app/builder` → `page = "builder"` → loads `pages/builder.js`.

### Step 3: Your Entry Point JS

```tsx
// src/main/webapp/pages/builder/index.tsx
import layout from '@splunk/react-page';
import UrcBuilder from '@splunk/urc-builder';

getUserTheme().then((theme) => {
    layout(
        <StyledContainer>
            <UrcBuilder />
        </StyledContainer>,
        { theme }
    );
});
```

`@splunk/react-page` is a helper that:
1. Creates a root `<div>` in the DOM
2. Wraps your component in `SplunkThemeProvider`
3. Calls `ReactDOM.createRoot().render()`

### The full chain visualized:

```
User clicks "Builder" in Splunk nav
         │
         ▼
Splunk reads builder.xml
         │ template="urc_app:/templates/builder.html"
         ▼
Splunk renders builder.html (Mako template)
         │ Injects __splunkd_partials__, loads config
         │ Adds <script src="pages/builder.js">
         ▼
Browser loads builder.js (YOUR WEBPACK BUNDLE)
         │
         ▼
builder/index.tsx executes
         │ import UrcBuilder from '@splunk/urc-builder'
         │ (already bundled inside builder.js by webpack)
         ▼
getUserTheme() fetches theme from Splunk REST API
         │ Uses __splunkd_partials__ to know the API endpoint
         ▼
layout(<UrcBuilder />) renders into DOM
         │
         ▼
React takes over, component is interactive
```

---

## 8. The Three Webpack Configs Per Package

Each component library package has THREE webpack configs for three different contexts. (The app wrapper only uses its production config — the demo configs are scaffolded but not used for UCC overlay work.)

```
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  webpack.config.js              PRODUCTION BUILD             │
│  "Build the real thing"         Output: distributable files  │
│                                 Used by: yarn build          │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  demo/webpack.splunkapp.config.js    SPLUNK DEV MODE         │
│  "Build for local Splunk"            Output: demo/splunk-app │
│                                      Used by: yarn start:app │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  demo/webpack.standalone.config.js   STANDALONE DEV MODE     │
│  "Build for localhost:8080"          Output: in-memory (HMR) │
│                                      Used by: yarn start:demo│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Why three?

**Production** (`webpack.config.js`): Builds the final artifact. For `urc-app`, dynamically discovers pages. For components, builds as a library.

**Splunk dev** (`webpack.splunkapp.config.js`): Builds `demo.tsx` into a Splunk app you can symlink into `$SPLUNK_HOME/etc/apps/`. Lets you test inside real Splunk with all the REST APIs available.

**Standalone dev** (`webpack.standalone.config.js`): Builds `demo.tsx` into a plain webpage served by `webpack-dev-server` on port 8080. No Splunk needed. Uses `HtmlWebpackPlugin` to generate the HTML:

```js
// demo/webpack.standalone.config.js
plugins: [
    new HtmlWebpackPlugin({
        template: path.join(__dirname, 'standalone/index.html'),
        //                                ↑ A minimal HTML file:
        // <body><div id="main-component-container"></div></body>
    }),
],
```

---

## 9. Component vs Application: Two Different Build Strategies

This is a crucial distinction in the Splunk React ecosystem:

### Component build (urc-builder, test-page):

```js
// packages/urc-builder/webpack.config.js
const baseComponentConfig = require('@splunk/webpack-configs/component.config').default;

module.exports = webpackMerge(baseComponentConfig, {
    entry: { UrcBuilder: path.join(__dirname, 'src/UrcBuilder.tsx') },
    output: { path: path.join(__dirname) },  // outputs to package root
});
```

**Component config** (from `@splunk/webpack-configs/component.config.js`):
```js
{
    devtool: false,
    externals: /^[^/.][a-zA-Z\-0-9./]+$/,  // <-- THE KEY DIFFERENCE
    output: {
        libraryTarget: 'commonjs2',  // module.exports = ...
    },
    optimization: {
        minimize: true,
        minimizer: [terserPlugin]  // Beautified output, not minified
    }
}
```

**`externals: /^[^/.][a-zA-Z\-0-9./]+$/`** — This regex says: "anything imported with a bare module name (like `react`, `@splunk/react-ui/Button`) should NOT be bundled. Leave the `require()` call in the output."

This means `UrcBuilder.js` output looks like:

```js
// UrcBuilder.js (component build output — NOT a full bundle)
var React = require('react');                    // <-- NOT bundled, just a require
var Button = require('@splunk/react-ui/Button'); // <-- NOT bundled

module.exports = function UrcBuilder() { ... };
```

**Why?** Components are libraries. The final application will bundle everything. If the component also bundled React, you'd get two copies.

### Application build (urc-app):

```js
// packages/urc-app/webpack.config.js
const baseConfig = require('@splunk/webpack-configs/base.config').default;  // <-- base, not component!

module.exports = webpackMerge(baseConfig, {
    entry: entries,
    output: {
        path: path.join(__dirname, 'stage/appserver/static/pages/'),
        filename: '[name].js',
    },
});
```

**Base config** has NO `externals`. Everything gets bundled. The output `builder.js` contains React, all Splunk UI components, your UrcBuilder, styled-components — everything the browser needs in one file.

### Visualized:

```
COMPONENT BUILD (urc-builder):
──────────────────────────────
UrcBuilder.tsx ──→ webpack (component.config) ──→ UrcBuilder.js
                                                   │
                        require('react')    ←──────┘  (external, not bundled)
                        require('@splunk/react-ui/Button')  (external)
                        actual component code               (bundled)

APPLICATION BUILD (urc-app):
────────────────────────────
builder/index.tsx ──→ webpack (base.config) ──→ builder.js
    │                                              │
    ├── import UrcBuilder ←── follows symlink      │ Contains ALL of:
    │       └── import React                       │  - React runtime
    │       └── import Button                      │  - Splunk UI components
    ├── import @splunk/react-page                  │  - UrcBuilder code
    └── import @splunk/themes                      │  - Theme provider
                                                   │  - Everything
                                                   │
                                          ONE BIG FILE for the browser
```

---

## 10. The Full Build Flow: What Happens When You Run `yarn build`

```
$ yarn build
  → yarn workspaces run build
    → runs "build" script in each package

┌─ packages/test-page ──────────────────────────────────────────┐
│  "build": "node bin/build.js build && yarn types:build"       │
│                                                                │
│  Step 1: bin/build.js detects OS (win32 vs unix)               │
│  Step 2: Sets NODE_ENV=production                              │
│  Step 3: Runs: webpack --mode=production                       │
│          Config: webpack.config.js (component.config base)     │
│          Input:  src/TestPage.tsx                               │
│          Output: TestPage.js (externalized requires)           │
│  Step 4: Runs: tsc (TypeScript compiler)                       │
│          Config: tsconfig.json                                 │
│          Input:  src/**/*.tsx                                   │
│          Output: types/TestPage.d.ts (type declarations only)  │
└────────────────────────────────────────────────────────────────┘

┌─ packages/urc-builder ────────────────────────────────────────┐
│  Same as test-page: webpack (component) + tsc                  │
│  Input:  src/UrcBuilder.tsx                                    │
│  Output: UrcBuilder.js + types/UrcBuilder.d.ts                 │
└────────────────────────────────────────────────────────────────┘

┌─ packages/urc-app ────────────────────────────────────────────┐
│  "build": "node bin/build.js build"  (NO types:build!)         │
│                                                                │
│  Step 1: Sets NODE_ENV=production                              │
│  Step 2: Runs: webpack --mode=production                       │
│          Config: webpack.config.js (base.config — FULL bundle) │
│          Entry discovery: reads src/main/webapp/pages/*         │
│            → finds "builder" directory                          │
│            → entry: { builder: '.../pages/builder/index.tsx' }  │
│                                                                │
│  Step 3: Webpack resolves imports:                              │
│    builder/index.tsx                                           │
│      → import UrcBuilder from '@splunk/urc-builder'            │
│        → symlink → packages/urc-builder/src/index.ts           │
│          → import UrcBuilder from './UrcBuilder'               │
│            → UrcBuilder.tsx (Babel transforms it)               │
│              → import Button from '@splunk/react-ui/Button'    │
│                → node_modules/@splunk/react-ui/...             │
│                  → (and ALL of its dependencies too)            │
│                                                                │
│  Step 4: CopyWebpackPlugin copies:                             │
│    src/main/resources/splunk/**  →  stage/**                   │
│    (XML views, HTML templates, app.conf, nav)                  │
│                                                                │
│  Final output in stage/:                                       │
│    stage/appserver/static/pages/builder.js  (the bundle)       │
│    stage/appserver/templates/builder.html                      │
│    stage/default/data/ui/views/builder.xml                     │
│    stage/default/app.conf                                      │
└────────────────────────────────────────────────────────────────┘
```

---

## 11. ESLint, Prettier, Stylelint: The Quality Tools

These three tools check code quality but at different levels:

```
┌──────────────┬──────────────────────────────────────────────────┐
│   Prettier   │  Formatting ONLY. Tabs, quotes, line length.     │
│              │  Has zero opinions about correctness.             │
│              │  Config: .prettierrc                              │
│              │  Rule: printWidth: 100, singleQuote: true         │
├──────────────┼──────────────────────────────────────────────────┤
│   ESLint     │  Code quality. Finds bugs, enforces patterns.    │
│              │  "You used `var` instead of `const`"              │
│              │  "This import doesn't exist"                      │
│              │  "React hooks must follow rules of hooks"         │
│              │  Config: .eslintrc.js                             │
├──────────────┼──────────────────────────────────────────────────┤
│  Stylelint   │  CSS quality. For styled-components CSS-in-JS.   │
│              │  "Unknown CSS property"                           │
│              │  "Don't use !important"                           │
│              │  Config: stylelint.config.js                      │
└──────────────┴──────────────────────────────────────────────────┘
```

### How ESLint config inheritance works:

```js
// packages/urc-builder/.eslintrc.js
module.exports = {
    parser: '@typescript-eslint/parser',         // Use TS parser, not default JS parser
    plugins: ['@typescript-eslint'],             // Load TS-specific rules
    extends: [
        '@splunk/eslint-config/base',            // Splunk's standard rules
        '@splunk/eslint-config/browser-prettier'  // Turn off rules that conflict with Prettier
    ],
    rules: {
        'react/jsx-filename-extension': ['error', { extensions: ['.tsx', '.jsx'] }],
    },
};
```

The `extends` chain works like CSS cascading:
```
@splunk/eslint-config/base  →  adds hundreds of rules (imports Airbnb config)
     ↓ merged with
@splunk/eslint-config/browser-prettier  →  disables rules that Prettier handles
     ↓ merged with
your "rules" block  →  overrides specific rules
```

### Why `eslint-config-prettier` exists:

Without it, ESLint says "use double quotes" while Prettier says "use single quotes." They fight. `eslint-config-prettier` disables all ESLint rules that Prettier already handles, so each tool stays in its lane.

---

## 12. Jest: The Test Runner

```js
// packages/urc-builder/jest.config.js
module.exports = {
    testMatch: ['**/*.unit.[jt]s?(x)'],
    //          Matches: Foo.unit.ts, Bar.unit.tsx, Baz.unit.js
    testEnvironment: 'jsdom',
    //              Simulates a browser DOM in Node.js
    setupFilesAfterSetup: ['<rootDir>/unit-test-setup-testing-library.ts'],
    //                     Runs before each test file
};
```

**`testEnvironment: 'jsdom'`**: Jest runs in Node.js, which has no `document` or `window`. `jsdom` fakes an entire browser environment so React can render components.

**`testMatch: ['**/*.unit.[jt]s?(x)']`**: Jest only runs files matching this glob. The naming convention `Foo.unit.tsx` keeps tests co-located with source but clearly identified.

**How Jest uses Babel**: Jest has its own Babel integration. When it loads a `.tsx` test file, it uses Babel to transform it (same `@splunk/babel-preset`), then runs it. It does NOT use webpack — each file is transformed individually.

```
Webpack world:                    Jest world:
─────────────                     ──────────
Many files → ONE bundle           Each test file transformed INDIVIDUALLY
import resolution via webpack     import resolution via Jest's module system
Runs in browser                   Runs in Node.js + jsdom
```

---

## 13. Dependencies vs DevDependencies vs PeerDependencies

Your `package.json` has three dependency sections. Here's what each means:

```json
{
  "dependencies": {
    "@splunk/react-ui": "^5.9.0",       // Needed at runtime
    "@splunk/themes": "^1.6.0"          // Part of the component's API
  },
  "devDependencies": {
    "webpack": "^5.88.2",               // Only needed to BUILD
    "jest": "^30.1.3",                  // Only needed to TEST
    "eslint": "^8.57.1",               // Only needed to LINT
    "typescript": "^5.8.3",            // Only needed for type-checking
    "@types/react": "^18.2.0"          // Only needed for type-checking
  },
  "peerDependencies": {
    "react": "^16.8.0 || ^17.0.0 || ^18.0.0",   // YOU must provide this
    "styled-components": "^5.3.10"                 // YOU must provide this
  }
}
```

### The mental model:

```
dependencies:      "I need these to work"
                   Installed automatically when someone installs your package

devDependencies:   "I need these to develop/build/test, but not to run"
                   NOT installed when someone else installs your package

peerDependencies:  "I need these but I don't want to bundle my own copy —
                    the person using me should provide them"
                   Prevents duplicate React/styled-components in the bundle
```

### Why React is a peerDependency:

```
WITHOUT peerDependencies:              WITH peerDependencies:
─────────────────────────              ──────────────────────

urc-app/node_modules/                  urc-app/node_modules/
├── react@18.2.0  (copy 1)            ├── react@18.2.0  (THE ONLY COPY)
└── @splunk/urc-builder/              └── @splunk/urc-builder/
    └── node_modules/                     └── (no react here — uses parent's)
        └── react@18.2.0  (copy 2!)

Two Reacts = broken hooks,             One React = everything works
broken context, subtle bugs
```

---

## 14. The Scaffolding Pipeline: From Nothing to Running App

You never write all those config files by hand. `@splunk/create` generates them, and your custom scripts automate the process.

### Layer 1: `@splunk/create` — Splunk's Official Scaffolder

An interactive CLI tool that generates boilerplate. It has three generators:

| Generator | Creates | Used by |
|-----------|---------|---------|
| `ReactSplunkApp` | App wrapper (package.json, webpack, Splunk resources) | `task react:create` |
| `ReactSplunkPage` | Page entry point (pages/X/index.tsx, view XML, template HTML) | `task react:add-page` |
| `ReactComponent` | Component library (src/Component.tsx, tests, webpack) | `task react:add-component` |

When you run `npx @splunk/create`, it fires **inquirer prompts** asking for names, types, etc. Problem: this doesn't work in CI/CD or automated scripts.

### Layer 2: `splunk-create-noninteractive.mjs` — Your Prompt Bypass

This script calls the same generators **programmatically**, skipping inquirer:

```js
// react/bin/splunk-create-noninteractive.mjs
import { create as createMemFs } from 'mem-fs';
import { create as createEditor } from 'mem-fs-editor';

const store = createMemFs();
const fsStore = createEditor(store);

// Call the generator directly with options — no prompts!
const { default: ReactSplunkApp } = await import('@splunk/create/lib/ReactSplunkAppGenerator.js');
await ReactSplunkApp(fsStore, { appName: name });

// Write generated files (skip existing unless --force)
await fsStore.commit();
```

**How `mem-fs` works**: Instead of writing directly to disk, generators write to an in-memory filesystem. At the end, `commit()` flushes everything to disk. This lets the script check for existing files and skip them.

```
Usage:
  node bin/splunk-create-noninteractive.mjs app       UrcApp
  node bin/splunk-create-noninteractive.mjs page      Builder  --app urc-app
  node bin/splunk-create-noninteractive.mjs component UrcBuilder
```

### Layer 3: `adapt-for-ucc.mjs` — The 6 Mechanical Transformations

`@splunk/create` scaffolds a **standalone** React Splunk app. But you need a React page that **merges into a UCC app**. These are different:

```
STANDALONE APP (what @splunk/create gives you):
─────────────────────────────────────────────────
- Uses component.config → exports a library
- Template path references its OWN app name
- No awareness of UCC's configuration/search pages
- No CopyWebpackPlugin (doesn't need stage/)

UCC OVERLAY APP (what you actually need):
─────────────────────────────────────────
- Uses base.config → bundles everything into page JS
- Template path references the UCC APP name (underscore version)
- nav.xml includes UCC's configuration + search views
- CopyWebpackPlugin copies Splunk resources to stage/
- stage/ output merges into UCC's build output
```

The `adapt-for-ucc.mjs` script applies 6 file transformations:

```
┌────┬──────────────────────────────┬────────────────────────────────────────────┐
│  # │  File                        │  What Changes                              │
├────┼──────────────────────────────┼────────────────────────────────────────────┤
│  1 │ webpack.config.js            │ component.config → base.config             │
│    │                              │ + CopyWebpackPlugin + dynamic page entry   │
│    │                              │ + output to stage/appserver/static/pages/  │
├────┼──────────────────────────────┼────────────────────────────────────────────┤
│  2 │ package.json                 │ + @splunk/react-page dependency            │
│    │                              │ + @splunk/urc-builder dependency           │
│    │                              │ + copy-webpack-plugin devDependency        │
│    │                              │ - yarn types:build from build script       │
├────┼──────────────────────────────┼────────────────────────────────────────────┤
│  3 │ pages/builder/index.tsx      │ Import from @splunk/urc-builder            │
│    │                              │ Use layout() from @splunk/react-page       │
│    │                              │ (was: standalone component render)          │
├────┼──────────────────────────────┼────────────────────────────────────────────┤
│  4 │ nav/default.xml              │ + <view name="configuration"/>             │
│    │                              │ + <view name="search"/>                    │
│    │                              │ (UCC's pages coexist with your React page) │
├────┼──────────────────────────────┼────────────────────────────────────────────┤
│  5 │ views/builder.xml            │ template="urc_app:/templates/builder.html" │
│    │                              │ (references UCC app name, not React pkg)   │
├────┼──────────────────────────────┼────────────────────────────────────────────┤
│  6 │ templates/builder.html       │ page_path = "/static/app/urc_app/pages/…" │
│    │                              │ (JS is served from UCC app's static dir)   │
└────┴──────────────────────────────┴────────────────────────────────────────────┘
```

**Why transformations 5 and 6 matter:** At runtime, Splunk serves files from `$SPLUNK_HOME/etc/apps/urc_app/`. The template and view must reference `urc_app` (the UCC app's underscore name), not `urc-app` (the React package name). The React build output gets physically copied into the UCC output directory.

---

## 15. The UCC Overlay: How React Merges Into a UCC App

This is the "last mile" that connects everything:

```
UCC Build (ucc-gen build):                React Build (yarn build):
───────────────────────                    ─────────────────────────
ucc/urc_app/globalConfig.json              react/packages/urc-app/
         │                                              │
         ▼                                              ▼
ucc/output/urc_app/                        react/packages/urc-app/stage/
├── appserver/                             ├── appserver/
│   ├── static/js/build/                   │   ├── static/pages/
│   │   └── entry_page.js  (UCC UI)       │   │   └── builder.js  (YOUR React UI)
│   └── templates/                         │   └── templates/
│       └── base.html      (UCC template)  │       └── builder.html (YOUR template)
├── bin/                                   ├── default/
│   ├── import_declare_test.py             │   ├── data/ui/views/builder.xml
│   └── urc_app_rh_*.py                   │   └── data/ui/nav/default.xml
├── default/                               └── ...
│   ├── app.conf
│   ├── inputs.conf.spec
│   └── restmap.conf
└── lib/ (vendored Python)
         │                                              │
         └──────────────┬───────────────────────────────┘
                        │  MERGE (cp -r stage/* → output/)
                        ▼
         ucc/output/urc_app/ (FINAL APP)
         ├── appserver/
         │   ├── static/
         │   │   ├── js/build/entry_page.js  (UCC UI — untouched)
         │   │   └── pages/builder.js        (YOUR React UI — added)
         │   └── templates/
         │       ├── base.html               (UCC template — untouched)
         │       └── builder.html            (YOUR template — added)
         ├── default/
         │   ├── app.conf                    (from UCC)
         │   ├── data/ui/views/
         │   │   ├── configuration.xml       (from UCC)
         │   │   ├── inputs.xml              (from UCC)
         │   │   └── builder.xml             (from React — ADDED)
         │   └── data/ui/nav/default.xml     (from React — OVERRIDES UCC)
         ├── bin/ (from UCC)
         └── lib/ (from UCC)
```

**The merge is a simple file copy** — React's `stage/` output drops into the UCC output directory. There's no conflict because:
- UCC puts its JS in `appserver/static/js/build/`
- React puts its JS in `appserver/static/pages/`
- UCC's template is `base.html`, React's is `builder.html`
- The nav.xml from React **overrides** UCC's default to make `builder` the home page

---

## 16. The Taskfile: Orchestrating Everything

The `Taskfile.yml` ties all the tools together into a workflow you can actually use. Here's how the key tasks map to the tooling:

### The Scaffolding Flow (one-time setup):

```
task ucc:add-react APP_NAME=urc_app
│
├── 1. Derive names:
│       UCC_APP    = urc_app        (underscore, Python/Splunk convention)
│       REACT_APP  = urc-app        (kebab-case, Node/npm convention)
│       COMPONENT  = urc-builder    (component library)
│       PAGE       = builder        (Splunk view name)
│
├── 2. node bin/splunk-create-noninteractive.mjs component UrcBuilder
│       → creates packages/urc-builder/ (component library)
│
├── 3. node bin/splunk-create-noninteractive.mjs app UrcApp
│       → creates packages/urc-app/ (Splunk app wrapper)
│
├── 4. node bin/adapt-for-ucc.mjs --app-pkg urc-app --ucc-app urc_app ...
│       → applies 6 transformations (standalone → UCC overlay)
│
├── 5. yarn install (resolves workspace symlinks)
│
└── 6. yarn workspace @splunk/urc-builder build
       yarn workspace @splunk/urc-app build
       → webpack builds, stage/ populated
```

### The Dev Loop (daily work):

```
task ucc:dev
│
├── task ucc:build
│   └── ucc-gen build --source ucc/urc_app -o ucc/output/
│       (generates Python, conf files, UCC UI)
│
├── task ucc:link
│   └── docker exec splunk-dev ln -s /opt/splunk/ucc/output/urc_app \
│                                     /opt/splunk/etc/apps/urc_app
│       (symlinks build output into running Splunk)
│
└── task dev:refresh
    └── docker exec splunk-dev splunk restart splunkweb
        (Splunk picks up new/changed files)
```

### React Dev (with hot rebuild):

For **standalone** React apps (not UCC overlay), `react:link` symlinks `stage/` into Splunk and `react:start` runs webpack in watch mode:

```
task react:link      (once — symlinks stage/ into Splunk)
task react:start     (ongoing — webpack --watch, rebuilds on save)
```

For **UCC overlay** apps, use the full build pipeline instead — the React output merges into the UCC build:

```
task ucc:dev         (UCC build + link + refresh — includes React merge)
```

During active React UI development on a UCC app, you can run webpack watch alongside:

```
cd react && yarn workspace @splunk/urc-app start
# (rebuilds stage/ on save — still need ucc:merge-ui to see changes in Splunk)
```

### The Packaging Flow (for staging/production):

```
task ucc:package
│
├── ucc-gen build        (Python + UCC UI)
├── ucc-gen package      (creates .tar.gz)
└── Output: splunk/stage/urc_app.tgz

task react:package
│
├── yarn build           (webpack production build)
├── tar -chzf            (packages stage/ as .tgz)
└── Output: splunk/stage/urc-app.tgz

task stage:deploy
│
├── task stage:package   (builds both tgz files)
├── task stage:up        (starts staging Splunk container)
└── task stage:install   (installs tgz files into staging)
```

### How APP_NAME flows through the system:

```
.env:  APP_NAME=urc_app

Taskfile reads .env
    │
    ├── ucc:resolve  → UCC_ADDON=urc_app,  UCC_PATH=ucc/urc_app
    ├── react:resolve → REACT_APP_NAME=urc-app, REACT_APP_PATH=react/packages/urc-app
    │
    ├── ucc:add-react derives:
    │   UCC_APP     = urc_app       (from APP_NAME)
    │   REACT_APP   = urc-app       (tr '_' '-')
    │   COMPONENT   = urc-builder   (replace -app with -builder)
    │   PAGE        = builder        (default, or PAGE_NAME override)
    │
    └── adapt-for-ucc.mjs uses:
        --ucc-app urc_app           → goes in HTML template paths, view XML
        --app-pkg urc-app           → React package directory name
        --component-pkg urc-builder → import path in index.tsx
        --page builder              → directory name, XML view name
```

---

## 17. Mental Model: The Complete Pipeline Diagram

```
                        YOUR MONOREPO
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  Taskfile.yml                    .env (APP_NAME=urc_app)                 │
│  ════════════                    ════════════════════════                 │
│  Orchestrates everything         Single source of truth for app name     │
│                                                                          │
│  ┌─ ucc/ ──────────────────────┐  ┌─ react/ ────────────────────────┐   │
│  │                              │  │                                  │   │
│  │  urc_app/                    │  │  bin/                            │   │
│  │  ├── globalConfig.json       │  │  ├── splunk-create-noninteractive│   │
│  │  └── package/                │  │  │   (scaffolds without prompts) │   │
│  │      ├── bin/ (Python)       │  │  └── adapt-for-ucc.mjs          │   │
│  │      └── lib/ (vendored)     │  │      (6 transformations)         │   │
│  │          │                   │  │                                  │   │
│  │          ▼                   │  │  packages/                       │   │
│  │  ucc-gen build               │  │  ├── urc-builder/ (COMPONENT)   │   │
│  │          │                   │  │  │   src/UrcBuilder.tsx          │   │
│  │          ▼                   │  │  │       │                       │   │
│  │  output/urc_app/             │  │  │       ▼ webpack (component)   │   │
│  │  ├── bin/                    │  │  │   UrcBuilder.js (externals)   │   │
│  │  ├── lib/                    │  │  │                               │   │
│  │  ├── default/                │  │  └── urc-app/ (APPLICATION)     │   │
│  │  ├── appserver/static/js/    │  │      src/main/webapp/pages/     │   │
│  │  │   └── build/entry_page.js │  │      builder/index.tsx          │   │
│  │  └── appserver/templates/    │  │          │ imports urc-builder   │   │
│  │      └── base.html           │  │          ▼ webpack (base)        │   │
│  │                              │  │      stage/                      │   │
│  │                              │  │      ├── appserver/static/pages/ │   │
│  │                              │  │      │   └── builder.js (BUNDLE)│   │
│  │                              │  │      ├── appserver/templates/    │   │
│  │                              │  │      │   └── builder.html       │   │
│  │                              │  │      └── default/data/ui/       │   │
│  │                              │  │          ├── views/builder.xml   │   │
│  └──────────────┬───────────────┘  │          └── nav/default.xml    │   │
│                 │                   └────────────────┬────────────────┘   │
│                 │                                    │                    │
│                 └───────────── MERGE ────────────────┘                    │
│                                │                                         │
│                                ▼                                         │
│                 output/urc_app/ (COMPLETE SPLUNK APP)                    │
│                 ├── bin/           Python handlers (UCC)                  │
│                 ├── lib/           Vendored deps (UCC)                   │
│                 ├── default/       Conf + views + nav (UCC + React)      │
│                 ├── appserver/                                            │
│                 │   ├── static/js/build/  UCC UI (auto-generated)        │
│                 │   ├── static/pages/     YOUR React UI (builder.js)     │
│                 │   └── templates/        Both templates coexist         │
│                 └── README/        Spec files                            │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                │
                    Deploy to Splunk
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  SPLUNK WEB SERVER                                                       │
│                                                                          │
│  User navigates to /app/urc_app/builder                                  │
│      ▼                                                                   │
│  Splunk reads builder.xml → template="urc_app:/templates/builder.html"   │
│      ▼                                                                   │
│  Renders builder.html (Mako) → injects __splunkd_partials__              │
│      ▼                                                                   │
│  Browser loads /static/app/urc_app/pages/builder.js                      │
│      ▼                                                                   │
│  builder.js calls getUserTheme() → fetches from Splunk REST API          │
│      ▼                                                                   │
│  layout(<UrcBuilder />) → React renders your component                   │
│                                                                          │
│  User navigates to /app/urc_app/configuration                            │
│      ▼                                                                   │
│  Splunk reads configuration.xml → UCC's base.html template              │
│      ▼                                                                   │
│  entry_page.js loads → UCC's auto-generated React UI renders             │
│  (accounts, inputs, logging — all from globalConfig.json)                │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference: "I want to..."

| I want to... | Command | What happens |
|---|---|---|
| **Scaffold everything from scratch** | `task ucc:add-react APP_NAME=x` | Scaffolds + adapts + builds |
| Install all deps | `cd react && yarn` | Installs to root node_modules, creates symlinks |
| Build everything | `cd react && yarn build` | Runs webpack in each package |
| Build one package | `yarn workspace @splunk/urc-builder run build` | Webpack + tsc for that package |
| Dev without Splunk | `yarn workspace @splunk/urc-builder run start:demo` | webpack-dev-server on :8080 |
| Dev with Splunk | `task react:link && task react:start` | Symlink + webpack --watch |
| Full UCC + React build | `task ucc:dev` | UCC build + link + refresh |
| Run tests | `cd react && yarn test` | Jest in all packages |
| Lint code | `cd react && yarn lint` | ESLint + Stylelint in all packages |
| Format code | `cd react && yarn format` | Prettier on all JS/CSS files |
| Check types | `yarn workspace @splunk/urc-builder run types:build` | tsc — reports errors, emits .d.ts |
| Add a new React page | `task react:add-page PAGE_NAME=x APP_NAME=y` | Scaffolds page + component |
| Add a component library | `task react:add-component COMPONENT_NAME=x` | Scaffolds component package |
| Package for staging | `task react:package` | Build + tar.gz to splunk/stage/ |

---

## Key Takeaways

1. **Most files are config, not code** — 14 config files exist so 3 source files can become 1 browser-ready bundle
2. **Babel compiles, TypeScript checks** — they process the same files for different purposes
3. **Webpack is the orchestrator** — it calls Babel via `babel-loader`, bundles imports, copies assets
4. **Component builds externalize deps** — they output a library with `require()` calls
5. **Application builds bundle everything** — they output one fat `.js` file for the browser
6. **Yarn workspaces** enable cross-package imports via symlinks, without publishing to npm
7. **Splunk's HTML template** is the bridge — it loads your webpack bundle into Splunk's authenticated context
8. **The Taskfile is your entry point** — you rarely need to know these details. `task ucc:add-react APP_NAME=x` scaffolds everything, `task ucc:dev` builds everything. The tools below are what those tasks invoke
7. **The UCC overlay** is a file merge — React's `stage/` output drops into UCC's `output/` directory
8. **`adapt-for-ucc.mjs`** applies 6 mechanical transformations so a standalone scaffold works as a UCC overlay
9. **The Taskfile** orchestrates scaffolding → building → linking → refreshing into single commands
10. **APP_NAME** is the single variable everything derives from — `urc_app` (Python) ↔ `urc-app` (Node) ↔ `UrcApp` (PascalCase)
11. **`webpack-merge`** is how Splunk provides sensible defaults you can override per-package
12. **Every config file exists because a tool needs it** — remove any one and that specific tool breaks
