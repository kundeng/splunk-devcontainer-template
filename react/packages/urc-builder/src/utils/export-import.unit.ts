import yaml from 'js-yaml';
import type { InputSummary } from '../types';
import {
    exportStreams,
    parseImportFile,
    detectConflicts,
    type ImportedStream,
} from './export-import';

/* ------------------------------------------------------------------ */
/*  Test fixtures                                                     */
/* ------------------------------------------------------------------ */

const MANIFEST_YAML = yaml.dump({
    type: 'DeclarativeSource',
    streams: [{ type: 'DeclarativeStream', name: 'events' }],
});

function makeInput(overrides: Partial<InputSummary> = {}): InputSummary {
    return {
        name: 'my-stream',
        account: 'acme-prod',
        baseUrl: 'https://api.example.com',
        interval: 300,
        index: 'main',
        sourcetype: 'urc:api:json',
        tags: 'prod,api',
        disabled: false,
        manifest: MANIFEST_YAML,
        debug: false,
        ...overrides,
    };
}

/* ------------------------------------------------------------------ */
/*  exportStreams                                                      */
/* ------------------------------------------------------------------ */
describe('exportStreams', () => {
    it('exports a single stream as valid YAML', () => {
        const result = exportStreams([makeInput()]);
        const parsed = yaml.load(result) as Record<string, unknown>;

        expect(parsed).toMatchObject({
            name: 'my-stream',
            account: 'acme-prod',
            base_url: 'https://api.example.com',
            interval: 300,
            index: 'main',
            sourcetype: 'urc:api:json',
            tags: 'prod,api',
            manifest: MANIFEST_YAML,
        });
    });

    it('exports multiple streams as multi-document YAML', () => {
        const inputs = [
            makeInput({ name: 'stream-a' }),
            makeInput({ name: 'stream-b' }),
        ];
        const result = exportStreams(inputs);

        // Should contain document separator
        expect(result).toContain('---');

        const docs = yaml.loadAll(result) as Record<string, unknown>[];
        expect(docs).toHaveLength(2);
        expect(docs[0].name).toBe('stream-a');
        expect(docs[1].name).toBe('stream-b');
    });

    it('does not include disabled, debug, or baseUrl fields', () => {
        const result = exportStreams([makeInput()]);
        const parsed = yaml.load(result) as Record<string, unknown>;

        expect(parsed).not.toHaveProperty('disabled');
        expect(parsed).not.toHaveProperty('debug');
        expect(parsed).not.toHaveProperty('baseUrl');
        // base_url (snake_case) IS included
        expect(parsed).toHaveProperty('base_url');
    });

    it('returns empty string for empty input array', () => {
        expect(exportStreams([])).toBe('');
    });
});

/* ------------------------------------------------------------------ */
/*  parseImportFile — YAML                                            */
/* ------------------------------------------------------------------ */
describe('parseImportFile (YAML)', () => {
    it('parses a single YAML document', () => {
        const doc = yaml.dump({ name: 'test', account: 'acme', base_url: 'https://x.com' });
        const result = parseImportFile(doc);

        expect(result).toHaveLength(1);
        expect(result[0].name).toBe('test');
        expect(result[0].account).toBe('acme');
        expect(result[0].base_url).toBe('https://x.com');
    });

    it('parses multi-document YAML', () => {
        const multi = [
            yaml.dump({ name: 'a', account: 'acme' }),
            yaml.dump({ name: 'b', account: 'acme' }),
        ].join('---\n');

        const result = parseImportFile(multi);
        expect(result).toHaveLength(2);
        expect(result[0].name).toBe('a');
        expect(result[1].name).toBe('b');
    });

    it('applies default values for missing fields', () => {
        const doc = yaml.dump({ name: 'minimal' });
        const result = parseImportFile(doc);

        expect(result[0].index).toBe('default');
        expect(result[0].sourcetype).toBe('urc:api:json');
        expect(result[0].interval).toBe(300);
        expect(result[0].tags).toBe('');
        expect(result[0].manifest).toBe('');
        expect(result[0].conflict).toBe('none');
        expect(result[0].resolution).toBe('create');
    });

    it('accepts baseUrl as an alias for base_url', () => {
        const doc = yaml.dump({ name: 'test', baseUrl: 'https://alias.com' });
        const result = parseImportFile(doc);
        expect(result[0].base_url).toBe('https://alias.com');
    });
});

/* ------------------------------------------------------------------ */
/*  parseImportFile — JSON                                            */
/* ------------------------------------------------------------------ */
describe('parseImportFile (JSON)', () => {
    it('parses a JSON array', () => {
        const json = JSON.stringify([
            { name: 'j1', account: 'a' },
            { name: 'j2', account: 'b' },
        ]);
        const result = parseImportFile(json);

        expect(result).toHaveLength(2);
        expect(result[0].name).toBe('j1');
        expect(result[1].name).toBe('j2');
    });

    it('wraps a single JSON object in an array', () => {
        const json = JSON.stringify({ name: 'single', account: 'a' });
        const result = parseImportFile(json);

        expect(result).toHaveLength(1);
        expect(result[0].name).toBe('single');
    });
});

/* ------------------------------------------------------------------ */
/*  parseImportFile — error handling                                  */
/* ------------------------------------------------------------------ */
describe('parseImportFile (error handling)', () => {
    it('returns empty array for completely invalid input', () => {
        // A bare string parses as a YAML scalar, not an object
        const result = parseImportFile('not valid at all }{}{');
        expect(result).toEqual([]);
    });

    it('skips null documents in multi-doc YAML', () => {
        const multi = '---\n\n---\nname: valid\n';
        const result = parseImportFile(multi);
        expect(result).toHaveLength(1);
        expect(result[0].name).toBe('valid');
    });
});

/* ------------------------------------------------------------------ */
/*  parseImportFile — XSS sanitisation                                */
/* ------------------------------------------------------------------ */
describe('parseImportFile (sanitisation)', () => {
    it('strips HTML tags from name', () => {
        const doc = yaml.dump({ name: '<script>alert("xss")</script>MyStream' });
        const result = parseImportFile(doc);
        expect(result[0].name).toBe('alert("xss")MyStream');
    });

    it('strips HTML tags from account', () => {
        const doc = yaml.dump({ name: 'ok', account: '<b>bold</b>' });
        const result = parseImportFile(doc);
        expect(result[0].account).toBe('bold');
    });

    it('strips HTML tags from tags field', () => {
        const doc = yaml.dump({ name: 'ok', tags: '<img src=x>prod' });
        const result = parseImportFile(doc);
        expect(result[0].tags).toBe('prod');
    });

    it('does not sanitise manifest (may contain angle brackets)', () => {
        const manifestWithBrackets = 'field_path: ["data", "<items>"]';
        const doc = yaml.dump({ name: 'ok', manifest: manifestWithBrackets });
        const result = parseImportFile(doc);
        expect(result[0].manifest).toBe(manifestWithBrackets);
    });
});

/* ------------------------------------------------------------------ */
/*  detectConflicts                                                   */
/* ------------------------------------------------------------------ */
describe('detectConflicts', () => {
    const existing = [makeInput({ name: 'alpha' }), makeInput({ name: 'beta' })];

    function makeImported(name: string): ImportedStream {
        return {
            name,
            account: 'acme',
            base_url: 'https://x.com',
            index: 'main',
            sourcetype: 'urc:api:json',
            interval: 300,
            tags: '',
            manifest: '',
            conflict: 'none',
            resolution: 'create',
        };
    }

    it('reports no conflicts when names are unique', () => {
        const imported = [makeImported('gamma'), makeImported('delta')];
        const report = detectConflicts(imported, existing);

        expect(report.clean).toHaveLength(2);
        expect(report.conflicts).toHaveLength(0);
    });

    it('reports conflicts for matching names', () => {
        const imported = [makeImported('alpha'), makeImported('gamma')];
        const report = detectConflicts(imported, existing);

        expect(report.clean).toHaveLength(1);
        expect(report.clean[0].name).toBe('gamma');

        expect(report.conflicts).toHaveLength(1);
        expect(report.conflicts[0].name).toBe('alpha');
        expect(report.conflicts[0].conflict).toBe('name');
    });

    it('reports all conflicts when all names match', () => {
        const imported = [makeImported('alpha'), makeImported('beta')];
        const report = detectConflicts(imported, existing);

        expect(report.clean).toHaveLength(0);
        expect(report.conflicts).toHaveLength(2);
    });

    it('handles empty imported list', () => {
        const report = detectConflicts([], existing);
        expect(report.clean).toEqual([]);
        expect(report.conflicts).toEqual([]);
    });

    it('handles empty existing list', () => {
        const imported = [makeImported('alpha')];
        const report = detectConflicts(imported, []);
        expect(report.clean).toHaveLength(1);
        expect(report.conflicts).toHaveLength(0);
    });
});

/* ------------------------------------------------------------------ */
/*  Round-trip: export then import                                    */
/* ------------------------------------------------------------------ */
describe('round-trip: export → import', () => {
    it('recovers equivalent data after export then import', () => {
        const original = [
            makeInput({ name: 'stream-1', tags: 'prod,api', interval: 60 }),
            makeInput({ name: 'stream-2', tags: 'dev', interval: 900 }),
        ];

        const exported = exportStreams(original);
        const imported = parseImportFile(exported);

        expect(imported).toHaveLength(2);

        // Check first stream
        expect(imported[0].name).toBe('stream-1');
        expect(imported[0].base_url).toBe('https://api.example.com');
        expect(imported[0].interval).toBe(60);
        expect(imported[0].tags).toBe('prod,api');
        expect(imported[0].manifest).toBe(MANIFEST_YAML);

        // Check second stream
        expect(imported[1].name).toBe('stream-2');
        expect(imported[1].interval).toBe(900);
        expect(imported[1].tags).toBe('dev');
    });

    it('preserves manifest content through round-trip', () => {
        const complexManifest = yaml.dump({
            type: 'DeclarativeSource',
            streams: [{
                type: 'DeclarativeStream',
                name: 'events',
                retriever: {
                    type: 'SimpleRetriever',
                    requester: { type: 'HttpRequester', url: '{{ config.base_url }}/events' },
                },
            }],
        });

        const original = [makeInput({ manifest: complexManifest })];
        const imported = parseImportFile(exportStreams(original));

        expect(imported[0].manifest).toBe(complexManifest);
    });
});
