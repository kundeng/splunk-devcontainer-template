/**
 * Splunk REST API client for URC Builder.
 *
 * Wraps @splunk/splunk-utils createRESTURL with fetch().
 * All UCC CRUD goes through standard AdminExternalHandler endpoints.
 */

import { createRESTURL } from '@splunk/splunk-utils/url';
import { getDefaultFetchInit } from '@splunk/splunk-utils/fetch';
import type { InputSummary, AccountSummary, InputPayload, TestResult, ValidationResult, StreamHealth } from '../types';

const JSON_PARAMS = 'output_mode=json&count=0';
async function splunkFetch(path: string, init?: RequestInit): Promise<any> {
    const url = createRESTURL(path);
    const separator = url.includes('?') ? '&' : '?';
    const fullUrl = `${url}${separator}${JSON_PARAMS}`;

    const defaults = getDefaultFetchInit();
    const resp = await fetch(fullUrl, {
        ...defaults,
        ...init,
        headers: {
            ...defaults.headers,
            'Content-Type': 'application/x-www-form-urlencoded',
            ...init?.headers,
        },
    });

    if (!resp.ok) {
        const text = await resp.text();
        let message = `HTTP ${resp.status}: ${resp.statusText}`;
        try {
            const json = JSON.parse(text);
            if (json.messages?.[0]?.text) {
                message = json.messages[0].text;
            }
        } catch {
            // Use default message
        }
        throw new Error(message);
    }

    return resp.json();
}

function formEncode(data: Record<string, string>): string {
    return Object.entries(data)
        .filter(([, v]) => v !== undefined && v !== null)
        .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
        .join('&');
}

/**
 * Parse Splunk's REST response envelope into a flat array.
 * Splunk returns { entry: [{ name, content: { ...fields } }] }
 */
function parseEntries<T>(response: any, mapper: (name: string, content: any) => T): T[] {
    const entries = response?.entry || [];
    return entries.map((e: any) => mapper(e.name, e.content || {}));
}

// ── Inputs (streams) ──

export async function listInputs(): Promise<InputSummary[]> {
    const data = await splunkFetch('data/inputs/urc_app_input');
    return parseEntries(data, (name, c) => ({
        name,
        account: c.account || '',
        baseUrl: c.base_url || '',
        interval: parseInt(c.interval, 10) || 300,
        index: c.index || 'default',
        sourcetype: c.sourcetype || 'urc:api:json',
        tags: c.tags || '',
        disabled: c.disabled === '1' || c.disabled === true,
        manifest: c.manifest || '',
        debug: c.debug === '1' || c.debug === true,
    }));
}

export async function getInput(name: string): Promise<InputSummary> {
    const data = await splunkFetch(`data/inputs/urc_app_input/${encodeURIComponent(name)}`);
    const entries = parseEntries<InputSummary>(data, (n, c) => ({
        name: n,
        account: c.account || '',
        baseUrl: c.base_url || '',
        interval: parseInt(c.interval, 10) || 300,
        index: c.index || 'default',
        sourcetype: c.sourcetype || 'urc:api:json',
        tags: c.tags || '',
        disabled: c.disabled === '1' || c.disabled === true,
        manifest: c.manifest || '',
        debug: c.debug === '1' || c.debug === true,
    }));
    if (entries.length === 0) {
        throw new Error(`Input '${name}' not found`);
    }
    return entries[0];
}

export async function createInput(payload: InputPayload): Promise<void> {
    await splunkFetch('data/inputs/urc_app_input', {
        method: 'POST',
        body: formEncode(payload as unknown as Record<string, string>),
    });
}

export async function updateInput(name: string, payload: Partial<InputPayload>): Promise<void> {
    await splunkFetch(`data/inputs/urc_app_input/${encodeURIComponent(name)}`, {
        method: 'POST',
        body: formEncode(payload as unknown as Record<string, string>),
    });
}

export async function deleteInput(name: string): Promise<void> {
    await splunkFetch(`data/inputs/urc_app_input/${encodeURIComponent(name)}`, {
        method: 'DELETE',
    });
}

export async function toggleInput(name: string, disabled: boolean): Promise<void> {
    await updateInput(name, { disabled: disabled ? '1' : '0' });
}

// ── Accounts ──

export async function listAccounts(): Promise<AccountSummary[]> {
    const data = await splunkFetch('urc_app_account');
    return parseEntries(data, (name, c) => ({
        name,
        authType: c.auth_type || 'none',
    }));
}

// ── Indexes ──

export async function listIndexes(): Promise<string[]> {
    const data = await splunkFetch('data/indexes');
    return parseEntries(data, (name) => name);
}

// ── Stream Health (KV store) ──

export async function getStreamHealth(): Promise<StreamHealth[]> {
    try {
        const url = createRESTURL('storage/collections/data/urc_stream_health');
        const defaults = getDefaultFetchInit();
        const resp = await fetch(`${url}?output_mode=json`, {
            ...defaults,
            headers: { ...defaults.headers },
        });

        if (!resp.ok) return []; // graceful degradation

        const entries: any[] = await resp.json();
        return entries.map((e: any) => ({
            name: e._key || e.stream_name || '',
            lastRun: e.last_run || null,
            lastStatus: e.last_status || null,
            lastError: e.last_error || null,
            lastRecordCount: parseInt(e.last_record_count, 10) || 0,
            errorCount24h: parseInt(e.error_count_24h, 10) || 0,
        }));
    } catch {
        return []; // KV store may not exist yet
    }
}

// ── Test Connection ──

export async function testConnection(
    manifest: string,
    accountName: string,
    baseUrl: string
): Promise<TestResult> {
    const data = await splunkFetch('urc_app/test_connection', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            manifest,
            account_name: accountName,
            base_url: baseUrl,
            mode: 'test',
        }),
    });

    const payload = typeof data.payload === 'string' ? JSON.parse(data.payload) : data;
    const records = payload.records || [];
    const validation: ValidationResult[] = payload.validation || [];

    const result: TestResult = {
        status: payload.status || 'error',
        message: payload.message,
        records,
        recordCount: payload.record_count || records.length,
        validationResults: validation,
        detectedFields: detectFields(records),
        timestampCandidates: detectTimestamps(records),
    };

    updateAccountHealth(accountName, payload.status === 'success' ? 'success' : 'error');

    return result;
}

// ── Account Health (session-level tracking) ──

const ACCOUNT_HEALTH_KEY = 'urc-account-health';

export interface AccountHealth {
    lastTestStatus: 'success' | 'error';
    lastTestTime: string; // ISO timestamp
}

export function getAccountHealthMap(): Record<string, AccountHealth> {
    try {
        const raw = sessionStorage.getItem(ACCOUNT_HEALTH_KEY);
        return raw ? JSON.parse(raw) : {};
    } catch {
        return {};
    }
}

export function updateAccountHealth(accountName: string, status: 'success' | 'error'): void {
    try {
        const map = getAccountHealthMap();
        map[accountName] = {
            lastTestStatus: status,
            lastTestTime: new Date().toISOString(),
        };
        sessionStorage.setItem(ACCOUNT_HEALTH_KEY, JSON.stringify(map));
    } catch {
        // ignore
    }
}

export async function validateManifest(manifest: string): Promise<ValidationResult[]> {
    const data = await splunkFetch('urc_app/test_connection', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            manifest,
            mode: 'validate',
        }),
    });

    const payload = typeof data.payload === 'string' ? JSON.parse(data.payload) : data;
    return payload.validation || [];
}

// ── Field & timestamp detection helpers ──

function detectFields(records: { data: Record<string, any> }[]): { name: string; type: string; sample: any }[] {
    const fieldMap = new Map<string, { type: string; sample: any }>();

    for (const rec of records) {
        const data = rec.data || rec;
        for (const [key, value] of Object.entries(data)) {
            if (!fieldMap.has(key)) {
                fieldMap.set(key, { type: inferType(value), sample: value });
            }
        }
    }

    return Array.from(fieldMap.entries()).map(([name, info]) => ({
        name,
        type: info.type,
        sample: info.sample,
    }));
}

function inferType(value: any): string {
    if (value === null || value === undefined) return 'null';
    if (typeof value === 'boolean') return 'boolean';
    if (typeof value === 'number') return 'number';
    if (Array.isArray(value)) return 'array';
    if (typeof value === 'object') return 'object';
    if (typeof value === 'string' && isDatetimeString(value)) return 'datetime';
    return 'string';
}

const TIMESTAMP_FIELD_PATTERNS = [
    /_at$/, /_date$/, /_time$/, /timestamp/, /^created/, /^updated/,
    /^modified/, /^date$/, /^time$/, /datetime/,
];

function isDatetimeString(value: string): boolean {
    if (!value || value.length < 8 || value.length > 40) return false;
    const parsed = Date.parse(value);
    return !isNaN(parsed);
}

function detectTimestamps(records: { data: Record<string, any> }[]): { field: string; sampleValue: string; parsedPreview: string }[] {
    if (records.length === 0) return [];

    const candidates: { field: string; sampleValue: string; parsedPreview: string }[] = [];
    const firstRecord = records[0].data || records[0];

    for (const [key, value] of Object.entries(firstRecord)) {
        if (value === null || value === undefined) continue;
        const strVal = String(value);

        // Check field name heuristic
        const nameMatch = TIMESTAMP_FIELD_PATTERNS.some((p) => p.test(key.toLowerCase()));

        // Check value heuristic
        let valueMatch = false;
        let parsedPreview = '';

        if (typeof value === 'string' && isDatetimeString(value)) {
            valueMatch = true;
            parsedPreview = new Date(value).toISOString();
        } else if (typeof value === 'number' && value > 946684800) {
            // Epoch seconds (after year 2000)
            valueMatch = true;
            const ms = value > 1e12 ? value : value * 1000;
            parsedPreview = new Date(ms).toISOString();
        }

        if (nameMatch || valueMatch) {
            candidates.push({
                field: key,
                sampleValue: strVal,
                parsedPreview: parsedPreview || 'Unable to parse',
            });
        }
    }

    return candidates;
}
