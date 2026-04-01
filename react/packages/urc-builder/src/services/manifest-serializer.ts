/**
 * Bidirectional conversion between BuilderState form state and
 * Airbyte declarative manifest YAML.
 *
 * The manifest YAML is the "source of truth" stored in Splunk's inputs.conf.
 * The form state is the UI representation used by the builder tabs.
 *
 * Key mapping rules:
 * - account, baseUrl, index, sourcetype, tags, interval, debug, enabled
 *   are UCC input fields — NOT part of the manifest YAML
 * - The manifest contains: type, streams[], with retriever, pagination,
 *   incremental_sync, transformations, event_timestamp
 */

import yaml from 'js-yaml';
import type { BuilderState, StreamConfig } from '../types';

// ── Default empty stream config ──

const DEFAULT_STREAM: StreamConfig = {
    name: 'default',
    retriever: {
        type: 'SimpleRetriever',
        requester: {
            type: 'HttpRequester',
            url: '',
            http_method: 'GET',
        },
        recordSelector: {
            type: 'RecordSelector',
            extractor: {
                type: 'DpathExtractor',
                field_path: [],
            },
        },
    },
};

// ── Default builder state ──

export const INITIAL_BUILDER_STATE: BuilderState = {
    mode: 'create',
    inputName: null,
    account: '',
    baseUrl: '',
    httpMethod: 'GET',
    streams: [{ ...DEFAULT_STREAM }],
    index: 'default',
    sourcetype: 'urc:api:json',
    sourceOverride: '',
    hostOverride: '',
    timestampField: null,
    timestampFormat: '',
    interval: 300,
    tags: '',
    enabled: true,
    debug: false,
    testResult: null,
    testLoading: false,
    validationResults: [],
    isDirty: false,
};

// ── Form state → Manifest YAML ──

export function formStateToManifest(state: BuilderState): string {
    const manifest: Record<string, any> = {
        type: 'DeclarativeSource',
        streams: state.streams.map((stream) => buildStreamDef(stream, state)),
    };

    return yaml.dump(manifest, {
        indent: 2,
        lineWidth: 120,
        noRefs: true,
        sortKeys: false,
    });
}

function buildStreamDef(stream: StreamConfig, state: BuilderState): Record<string, any> {
    const def: Record<string, any> = {
        type: 'DeclarativeStream',
        name: stream.name,
        retriever: buildRetriever(stream, state),
    };

    if (stream.incrementalSync && stream.incrementalSync.type) {
        def.incremental_sync = cleanEmpty(stream.incrementalSync);
    }

    if (stream.transformations && stream.transformations.length > 0) {
        def.transformations = stream.transformations
            .filter((t) => t && t.type)
            .map(cleanEmpty);
    }

    if (stream.eventTimestamp && stream.eventTimestamp.type) {
        def.event_timestamp = cleanEmpty(stream.eventTimestamp);
    }

    return def;
}

function buildRetriever(stream: StreamConfig, state: BuilderState): Record<string, any> {
    const retriever: Record<string, any> = {
        type: stream.retriever.type || 'SimpleRetriever',
        requester: {
            ...cleanEmpty(stream.retriever.requester),
            type: stream.retriever.requester.type || 'HttpRequester',
            url: stream.retriever.requester.url || "{{ config['base_url'] }}",
            http_method: state.httpMethod || 'GET',
        },
        record_selector: cleanEmpty(stream.retriever.recordSelector),
    };

    if (stream.retriever.paginator && stream.retriever.paginator.type) {
        retriever.paginator = cleanEmpty(stream.retriever.paginator);
    }

    return retriever;
}

// ── Manifest YAML → Form state ──

export function manifestToFormState(
    manifestYaml: string,
    defaults?: Partial<BuilderState>
): Partial<BuilderState> {
    let manifest: Record<string, any>;
    try {
        manifest = yaml.load(manifestYaml) as Record<string, any>;
    } catch (e) {
        return { ...defaults };
    }

    if (!manifest || typeof manifest !== 'object') {
        return { ...defaults };
    }

    const rawStreams = manifest.streams || [];
    const streams: StreamConfig[] = rawStreams.map(parseStream);

    // Infer HTTP method from first stream's requester
    const firstRequester = streams[0]?.retriever?.requester;
    const httpMethod = (firstRequester?.http_method || 'GET').toUpperCase() as 'GET' | 'POST';

    return {
        ...defaults,
        streams: streams.length > 0 ? streams : [{ ...DEFAULT_STREAM }],
        httpMethod,
        isDirty: false,
    };
}

function parseStream(raw: Record<string, any>): StreamConfig {
    const retriever = raw.retriever || {};
    const requester = retriever.requester || {};
    const recordSelector = retriever.record_selector || {};
    const paginator = retriever.paginator;

    return {
        name: raw.name || 'default',
        retriever: {
            type: retriever.type || 'SimpleRetriever',
            requester: {
                type: requester.type || 'HttpRequester',
                ...requester,
            },
            recordSelector: {
                type: recordSelector.type || 'RecordSelector',
                ...recordSelector,
            },
            ...(paginator ? { paginator } : {}),
        },
        incrementalSync: raw.incremental_sync || undefined,
        transformations: raw.transformations || undefined,
        eventTimestamp: raw.event_timestamp || undefined,
    };
}

// ── Helpers ──

/**
 * Remove keys with null, undefined, or empty string values.
 * Keeps false, 0, and empty arrays.
 */
function cleanEmpty(obj: Record<string, any>): Record<string, any> {
    const result: Record<string, any> = {};
    for (const [key, value] of Object.entries(obj)) {
        if (value === null || value === undefined || value === '') continue;
        if (typeof value === 'object' && !Array.isArray(value)) {
            const cleaned = cleanEmpty(value);
            if (Object.keys(cleaned).length > 0) {
                result[key] = cleaned;
            }
        } else {
            result[key] = value;
        }
    }
    return result;
}
