/**
 * Import/export utilities for URC stream configurations.
 *
 * Streams are exported as multi-document YAML (--- separator) and can be
 * imported from either YAML or JSON. Sensitive fields (account credentials)
 * are deliberately excluded from the export format.
 */

import yaml from 'js-yaml';
import type { InputSummary } from '../types';

// ── Exported types ──

export interface ExportedStream {
    name: string;
    account: string;
    base_url: string;
    index: string;
    sourcetype: string;
    interval: number;
    tags: string;
    manifest: string;
}

export interface ImportedStream extends ExportedStream {
    conflict: 'none' | 'name';
    resolution: 'create' | 'skip' | 'rename' | 'overwrite';
}

export interface ConflictReport {
    clean: ImportedStream[];
    conflicts: ImportedStream[];
}

// ── Export ──

/**
 * Export streams as multi-document YAML (--- separator).
 * Does NOT include sensitive fields like account credentials.
 */
export function exportStreams(inputs: InputSummary[]): string {
    const docs = inputs.map((input) => ({
        name: input.name,
        account: input.account,
        base_url: input.baseUrl,
        index: input.index,
        sourcetype: input.sourcetype,
        interval: input.interval,
        tags: input.tags,
        manifest: input.manifest,
    }));

    return docs
        .map((doc) => yaml.dump(doc, { indent: 2, lineWidth: 120, noRefs: true }))
        .join('---\n');
}

// ── Import ──

/** Strip HTML tags from a string value for XSS prevention. */
const sanitize = (s: string): string => s.replace(/<[^>]*>/g, '');

/**
 * Parse an import file (YAML or JSON).
 * Supports multi-document YAML and JSON arrays.
 */
export function parseImportFile(content: string): ImportedStream[] {
    let parsed: unknown[];

    // Try JSON first
    try {
        const json = JSON.parse(content);
        parsed = Array.isArray(json) ? json : [json];
    } catch {
        // Try multi-document YAML
        parsed = yaml.loadAll(content) as unknown[];
    }

    return parsed
        .filter((doc): doc is Record<string, unknown> => doc != null && typeof doc === 'object')
        .map((doc) => ({
            name: sanitize(String(doc.name || '')),
            account: sanitize(String(doc.account || '')),
            base_url: sanitize(String(doc.base_url || doc.baseUrl || '')),
            index: sanitize(String(doc.index || 'default')),
            sourcetype: sanitize(String(doc.sourcetype || 'urc:api:json')),
            interval: Number(doc.interval) || 300,
            tags: sanitize(String(doc.tags || '')),
            manifest: String(doc.manifest || ''),
            conflict: 'none' as const,
            resolution: 'create' as const,
        }));
}

// ── Conflict detection ──

/**
 * Detect name collisions between imported streams and existing ones.
 */
export function detectConflicts(imported: ImportedStream[], existing: InputSummary[]): ConflictReport {
    const existingNames = new Set(existing.map((e) => e.name));

    const clean: ImportedStream[] = [];
    const conflicts: ImportedStream[] = [];

    for (const stream of imported) {
        if (existingNames.has(stream.name)) {
            conflicts.push({ ...stream, conflict: 'name' });
        } else {
            clean.push(stream);
        }
    }

    return { clean, conflicts };
}
