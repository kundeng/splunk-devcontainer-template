/**
 * Pure validation utilities for URC Builder forms.
 * No React imports, no side effects.
 */

const URL_PATTERN = /^https?:\/\/[^\s]+$/;
const STREAM_NAME_PATTERN = /^[a-zA-Z0-9_-]+$/;
const BODY_METHODS = new Set(['POST', 'PUT', 'PATCH']);

export function isValidUrl(value: string): boolean {
    if (!value) return false;
    if (!URL_PATTERN.test(value)) return false;
    // Reject bare protocol with no host
    return value.replace(/^https?:\/\//, '').length > 0;
}

export function isValidStreamName(value: string): boolean {
    if (!value) return false;
    return STREAM_NAME_PATTERN.test(value);
}

export function isValidInterval(value: number): boolean {
    if (typeof value !== 'number' || Number.isNaN(value)) return false;
    return value >= 10 && value <= 86400;
}

export function isNameUnique(name: string, existing: string[]): boolean {
    return !existing.includes(name);
}

export interface FieldError {
    field: string;
    message: string;
}

export function getCrossFieldWarnings(state: any): FieldError[] {
    const warnings: FieldError[] = [];
    const streams: any[] = state.streams ?? [];

    // 1. HTTP method is POST/PUT/PATCH but no request_body
    if (BODY_METHODS.has(state.httpMethod)) {
        const hasBody = streams.some(
            (s) => s?.retriever?.requester?.request_body
        );
        if (!hasBody) {
            warnings.push({
                field: 'httpMethod',
                message: `${state.httpMethod} method selected but no request body configured`,
            });
        }
    }

    for (const stream of streams) {
        // 2. incrementalSync configured but no cursor_field
        const inc = stream?.incrementalSync;
        if (inc && !inc.cursor_field) {
            warnings.push({
                field: 'incrementalSync',
                message: 'Incremental sync configured but no cursor field is mapped',
            });
        }

        // 3. paginator configured but record_selector extractors empty
        const paginator = stream?.retriever?.paginator;
        if (paginator) {
            const fieldPath = stream?.retriever?.recordSelector?.extractor?.field_path;
            if (!fieldPath || fieldPath.length === 0) {
                warnings.push({
                    field: 'paginator',
                    message: 'Pagination configured but no record selector path is set',
                });
            }
        }
    }

    return warnings;
}
