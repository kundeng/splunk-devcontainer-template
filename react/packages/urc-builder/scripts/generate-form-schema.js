#!/usr/bin/env node
/**
 * generate-form-schema.js
 *
 * Build-time script that reads the Airbyte declarative component YAML schema
 * and URC extensions schema, then generates TypeScript form-schema definitions
 * for the URC Builder React app.
 *
 * Usage:  node scripts/generate-form-schema.js
 * Output: src/schema/form-schema.ts
 */

const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------
const AIRBYTE_SCHEMA_PATH = path.resolve(
    __dirname,
    '../../../../ucc/urc_app/schema/declarative_component_schema.yaml'
);
const URC_EXTENSIONS_PATH = path.resolve(
    __dirname,
    '../../../../ucc/urc_app/schema/urc_extensions_schema.yaml'
);
const OUTPUT_PATH = path.resolve(__dirname, '../src/schema/form-schema.ts');

// ---------------------------------------------------------------------------
// Supported types (registered in the Python engine's REGISTRY)
// ---------------------------------------------------------------------------
const SUPPORTED_TYPES = new Set([
    'NoAuth', 'ApiKeyAuthenticator', 'BearerAuthenticator',
    'BasicHttpAuthenticator', 'OAuthAuthenticator', 'SelectiveAuthenticator',
    'DigestHttpAuthenticator', 'JwtAuthenticator', 'SessionTokenAuthenticator',
    'NoPagination', 'DefaultPaginator', 'OffsetIncrement', 'PageIncrement',
    'CursorPagination', 'DpathExtractor', 'RecordSelector', 'RecordFilter',
    'SimpleRetriever', 'AsyncRetriever', 'HttpRequester',
    'AddFields', 'RemoveFields', 'KeysToLower', 'KeysToSnakeCase',
    'FlattenFields', 'DpathFlattenFields', 'KeysReplace', 'SchemaNormalization',
    'JsonDecoder', 'JsonlDecoder', 'CsvDecoder', 'XmlDecoder', 'GzipDecoder',
    'IterableDecoder', 'ZipfileDecoder',
    'ConstantBackoffStrategy', 'ExponentialBackoffStrategy', 'WaitTimeFromHeader',
    'HttpResponseFilter', 'DefaultErrorHandler', 'CompositeErrorHandler',
    'SubstreamPartitionRouter', 'ListPartitionRouter',
    'CartesianProductStreamSlicer',
    'APIBudget', 'TokenBucketRateLimiter', 'MovingWindowRateLimiterComponent',
    'CursorBasedTimestamp', 'FieldBasedTimestamp', 'FetchTimestamp',
]);

// ---------------------------------------------------------------------------
// Category assignment
// ---------------------------------------------------------------------------
const CATEGORY_EXACT = {
    // Paginators
    NoPagination: 'paginator',
    DefaultPaginator: 'paginator',
    OffsetIncrement: 'paginator',
    PageIncrement: 'paginator',
    CursorPagination: 'paginator',
    // Extractors
    DpathExtractor: 'extractor',
    RecordSelector: 'extractor',
    RecordFilter: 'extractor',
    // Transformations
    AddFields: 'transformation',
    RemoveFields: 'transformation',
    KeysToLower: 'transformation',
    KeysToSnakeCase: 'transformation',
    FlattenFields: 'transformation',
    DpathFlattenFields: 'transformation',
    KeysReplace: 'transformation',
    SchemaNormalization: 'transformation',
    // Error handler helpers
    HttpResponseFilter: 'error_handler',
    WaitTimeFromHeader: 'error_handler',
    // Partition routers
    CartesianProductStreamSlicer: 'partition_router',
    // Rate limiters
    APIBudget: 'rate_limiter',
    // Event timestamps (URC extensions)
    CursorBasedTimestamp: 'event_timestamp',
    FieldBasedTimestamp: 'event_timestamp',
    FetchTimestamp: 'event_timestamp',
    // Retriever / Requester
    SimpleRetriever: 'retriever',
    AsyncRetriever: 'retriever',
    HttpRequester: 'requester',
};

function categorize(typeName) {
    if (CATEGORY_EXACT[typeName]) return CATEGORY_EXACT[typeName];
    if (/Auth|Authenticator/.test(typeName)) return 'authenticator';
    if (/BackoffStrategy/.test(typeName)) return 'error_handler';
    if (/ErrorHandler/.test(typeName)) return 'error_handler';
    if (/Decoder/.test(typeName)) return 'decoder';
    if (/PartitionRouter/.test(typeName)) return 'partition_router';
    if (/RateLimiter/.test(typeName)) return 'rate_limiter';
    if (/Cursor/.test(typeName)) return 'cursor';
    return 'other';
}

// ---------------------------------------------------------------------------
// Category -> export constant mapping
// ---------------------------------------------------------------------------
const CATEGORY_TO_EXPORT = {
    authenticator: 'AUTHENTICATORS',
    paginator: 'PAGINATORS',
    extractor: 'EXTRACTORS',
    transformation: 'TRANSFORMATIONS',
    error_handler: 'ERROR_HANDLERS',
    cursor: 'INCREMENTAL_CURSORS',
    decoder: 'DECODERS',
    partition_router: 'PARTITION_ROUTERS',
    rate_limiter: 'RATE_LIMITERS',
    event_timestamp: 'EVENT_TIMESTAMPS',
    retriever: 'RETRIEVERS',
    requester: 'REQUESTERS',
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Turn snake_case or PascalCase into a human label */
function toLabel(s) {
    // PascalCase -> spaced
    let label = s.replace(/([a-z])([A-Z])/g, '$1 $2')
                  .replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2');
    // snake_case -> spaced
    label = label.replace(/_/g, ' ');
    // Title-case each word
    return label.replace(/\b\w/g, c => c.toUpperCase());
}

/** Determine the FormFieldDef type from a YAML property schema */
function resolveFieldType(prop) {
    // anyOf with multiple types — pick the first concrete one
    const effective = prop.anyOf ? prop.anyOf[0] : prop;

    if (prop.interpolation_context && !prop.enum) return 'template';
    if (prop.enum) return 'enum';

    const t = effective.type;
    if (Array.isArray(t)) {
        // e.g. ["integer", "string"] — pick first
        return resolveFieldType({ ...prop, type: t[0], anyOf: undefined });
    }
    if (t === 'string') {
        if (prop.interpolation_context) return 'template';
        if (prop.enum) return 'enum';
        return 'text';
    }
    if (t === 'integer' || t === 'number') return 'number';
    if (t === 'boolean') return 'boolean';
    if (t === 'object') return 'object';
    if (t === 'array') return 'array';
    // $ref without type
    if (effective.$ref) return 'object';
    return 'text';
}

/** Extract the type name from a $ref string like "#/definitions/Foo" */
function refName(ref) {
    if (!ref) return null;
    const parts = ref.split('/');
    return parts[parts.length - 1];
}

/** Resolve enum values — handles both enum arrays and const values */
function resolveEnum(prop) {
    if (prop.enum) {
        return prop.enum.map(v => ({ value: String(v), label: toLabel(String(v)) }));
    }
    if (prop.const !== undefined) {
        return [{ value: String(prop.const), label: toLabel(String(prop.const)) }];
    }
    return undefined;
}

/** Build a FormFieldDef from a property key and its YAML schema */
function buildField(key, prop, allDefs, depth = 0) {
    let helpText = (prop.description || '').replace(/\s+/g, ' ').trim();
    if (helpText.length > 150) helpText = helpText.slice(0, 147) + '...';

    const field = {
        key,
        label: prop.title || toLabel(key),
        helpText,
        type: resolveFieldType(prop),
        required: false, // caller sets this
    };

    if (prop.default !== undefined) field.defaultValue = prop.default;

    const enumVals = resolveEnum(prop);
    if (enumVals) field.enumValues = enumVals;

    if (prop.examples) {
        field.examples = prop.examples
            .map(e => (typeof e === 'string' ? e : JSON.stringify(e)))
            .slice(0, 3);
    }

    if (prop.interpolation_context) {
        field.interpolationContext = prop.interpolation_context;
    }

    // Resolve $ref one level deep (no recursion beyond depth 1)
    if (depth < 1) {
        const ref = prop.$ref || (prop.anyOf && prop.anyOf.find(a => a.$ref) || {}).$ref;
        if (ref && allDefs) {
            const refDef = allDefs[refName(ref)];
            if (refDef && refDef.properties) {
                field.type = 'object';
                field.nestedFields = extractFields(refDef, allDefs, true, depth + 1);
            }
        }
    }

    return field;
}

/** Extract all FormFieldDef[] from a definition's properties */
function extractFields(def, allDefs, resolveRefs = true, depth = 0) {
    const props = def.properties || {};
    const requiredSet = new Set(def.required || []);
    const fields = [];

    for (const [key, prop] of Object.entries(props)) {
        // Skip discriminator and internal fields
        if (key === 'type' || key === '$parameters') continue;
        // Skip deprecated fields
        if (prop.deprecated) continue;

        const field = buildField(key, prop, resolveRefs ? allDefs : null, depth);
        field.required = requiredSet.has(key);
        fields.push(field);
    }

    return fields;
}

/** Get the type discriminator value from a definition */
function getTypeDiscriminator(def) {
    const typeProp = def.properties && def.properties.type;
    if (!typeProp) return null;
    if (typeProp.enum && typeProp.enum.length === 1) return typeProp.enum[0];
    if (typeProp.const) return typeProp.const;
    return null;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
function main() {
    // Load schemas
    const airbyteSchema = yaml.load(fs.readFileSync(AIRBYTE_SCHEMA_PATH, 'utf8'));
    const urcSchema = yaml.load(fs.readFileSync(URC_EXTENSIONS_PATH, 'utf8'));

    const allDefs = {
        ...airbyteSchema.definitions,
        ...urcSchema.definitions,
    };

    // Process each definition
    const components = [];

    for (const [name, def] of Object.entries(allDefs)) {
        if (!def || def.type !== 'object') continue;
        const typeName = getTypeDiscriminator(def);
        if (!typeName) continue;

        const category = categorize(typeName);
        if (category === 'other') continue; // Skip non-form-relevant definitions

        components.push({
            type: typeName,
            category,
            label: def.title || toLabel(typeName),
            description: (def.description || '').replace(/\n/g, ' ').trim(),
            fields: extractFields(def, allDefs),
            supported: SUPPORTED_TYPES.has(typeName),
        });
    }

    // Sort: supported first, then alphabetically
    components.sort((a, b) => {
        if (a.supported !== b.supported) return a.supported ? -1 : 1;
        return a.type.localeCompare(b.type);
    });

    // Group by category
    const grouped = {};
    for (const cat of Object.keys(CATEGORY_TO_EXPORT)) {
        grouped[cat] = [];
    }
    for (const comp of components) {
        if (grouped[comp.category]) {
            grouped[comp.category].push(comp);
        }
    }

    // Generate TypeScript
    const lines = [];
    lines.push('// Auto-generated by scripts/generate-form-schema.js — DO NOT EDIT');
    lines.push('// Source: declarative_component_schema.yaml + urc_extensions_schema.yaml');
    lines.push(`// Generated: ${new Date().toISOString()}`);
    lines.push('');
    lines.push('export interface FormFieldDef {');
    lines.push('    key: string;');
    lines.push('    label: string;');
    lines.push('    helpText: string;');
    lines.push("    type: 'text' | 'number' | 'boolean' | 'enum' | 'object' | 'array' | 'template';");
    lines.push('    required: boolean;');
    lines.push('    defaultValue?: any;');
    lines.push('    enumValues?: { value: string; label: string }[];');
    lines.push('    examples?: string[];');
    lines.push('    nestedFields?: FormFieldDef[];');
    lines.push('    interpolationContext?: string[];');
    lines.push('}');
    lines.push('');
    lines.push('export interface ComponentFormDef {');
    lines.push('    type: string;');
    lines.push('    category: string;');
    lines.push('    label: string;');
    lines.push('    description: string;');
    lines.push('    fields: FormFieldDef[];');
    lines.push('    supported: boolean;');
    lines.push('}');
    lines.push('');

    // Emit each category
    for (const [cat, exportName] of Object.entries(CATEGORY_TO_EXPORT)) {
        const items = grouped[cat];
        lines.push(`export const ${exportName}: ComponentFormDef[] = ${serializeComponents(items)};`);
        lines.push('');
    }

    // ALL_COMPONENTS
    const allExports = Object.values(CATEGORY_TO_EXPORT);
    lines.push(`export const ALL_COMPONENTS: ComponentFormDef[] = [`);
    for (const e of allExports) {
        lines.push(`    ...${e},`);
    }
    lines.push('];');
    lines.push('');

    fs.mkdirSync(path.dirname(OUTPUT_PATH), { recursive: true });
    fs.writeFileSync(OUTPUT_PATH, lines.join('\n'));

    // Stats
    const supportedCount = components.filter(c => c.supported).length;
    console.log(`Generated ${OUTPUT_PATH}`);
    console.log(`  ${components.length} components (${supportedCount} supported)`);
    for (const [cat, exportName] of Object.entries(CATEGORY_TO_EXPORT)) {
        const items = grouped[cat];
        if (items.length > 0) {
            console.log(`  ${exportName}: ${items.length}`);
        }
    }
}

/** Serialize a ComponentFormDef[] to compact but readable TypeScript */
function serializeComponents(components) {
    if (components.length === 0) return '[]';
    const items = components.map(c => serializeComponent(c));
    return '[\n' + items.join(',\n') + ',\n]';
}

function serializeComponent(comp) {
    const lines = [];
    lines.push(`  { type: ${JSON.stringify(comp.type)}, category: ${JSON.stringify(comp.category)},`);
    lines.push(`    label: ${JSON.stringify(comp.label)}, supported: ${comp.supported},`);
    // Truncate description for output size
    const desc = comp.description.length > 200 ? comp.description.slice(0, 197) + '...' : comp.description;
    lines.push(`    description: ${JSON.stringify(desc)},`);
    lines.push(`    fields: ${serializeFields(comp.fields, 4)} }`);
    return lines.join('\n');
}

function serializeFields(fields, indent) {
    if (fields.length === 0) return '[]';
    const pad = ' '.repeat(indent);
    const items = fields.map(f => serializeField(f, indent + 2));
    return '[\n' + items.join(',\n') + ',\n' + pad + ']';
}

function serializeField(field, indent) {
    const pad = ' '.repeat(indent);
    // Build a compact single-object representation
    const parts = [];
    parts.push(`key: ${JSON.stringify(field.key)}`);
    parts.push(`label: ${JSON.stringify(field.label)}`);
    parts.push(`type: ${JSON.stringify(field.type)}`);
    parts.push(`required: ${field.required}`);
    parts.push(`helpText: ${JSON.stringify(field.helpText)}`);
    if (field.defaultValue !== undefined) {
        parts.push(`defaultValue: ${JSON.stringify(field.defaultValue)}`);
    }
    if (field.enumValues) {
        parts.push(`enumValues: ${JSON.stringify(field.enumValues)}`);
    }
    if (field.examples) {
        parts.push(`examples: ${JSON.stringify(field.examples)}`);
    }
    if (field.interpolationContext) {
        parts.push(`interpolationContext: ${JSON.stringify(field.interpolationContext)}`);
    }

    // Nested fields on separate lines
    if (field.nestedFields && field.nestedFields.length > 0) {
        const inner = serializeFields(field.nestedFields, indent + 2);
        return `${pad}{ ${parts.join(', ')},\n${pad}  nestedFields: ${inner} }`;
    }

    return `${pad}{ ${parts.join(', ')} }`;
}

main();
