import {
    isValidUrl,
    isValidStreamName,
    isValidInterval,
    isNameUnique,
    getCrossFieldWarnings,
    FieldError,
} from './validators';

// ── isValidUrl ──

describe('isValidUrl', () => {
    test('accepts http URL', () => {
        expect(isValidUrl('http://example.com')).toBe(true);
    });

    test('accepts https URL', () => {
        expect(isValidUrl('https://api.example.com/v1/data')).toBe(true);
    });

    test('accepts https with port', () => {
        expect(isValidUrl('https://localhost:8089/services')).toBe(true);
    });

    test('rejects empty string', () => {
        expect(isValidUrl('')).toBe(false);
    });

    test('rejects URL without protocol', () => {
        expect(isValidUrl('example.com')).toBe(false);
    });

    test('rejects URL with spaces', () => {
        expect(isValidUrl('https://exa mple.com')).toBe(false);
    });

    test('rejects ftp protocol', () => {
        expect(isValidUrl('ftp://files.example.com')).toBe(false);
    });

    test('rejects just a protocol', () => {
        expect(isValidUrl('https://')).toBe(false);
    });
});

// ── isValidStreamName ──

describe('isValidStreamName', () => {
    test('accepts alphanumeric', () => {
        expect(isValidStreamName('myStream1')).toBe(true);
    });

    test('accepts underscores', () => {
        expect(isValidStreamName('my_stream')).toBe(true);
    });

    test('accepts hyphens', () => {
        expect(isValidStreamName('my-stream')).toBe(true);
    });

    test('accepts mixed valid chars', () => {
        expect(isValidStreamName('Stream_01-test')).toBe(true);
    });

    test('rejects empty string', () => {
        expect(isValidStreamName('')).toBe(false);
    });

    test('rejects spaces', () => {
        expect(isValidStreamName('my stream')).toBe(false);
    });

    test('rejects dots', () => {
        expect(isValidStreamName('my.stream')).toBe(false);
    });

    test('rejects slashes', () => {
        expect(isValidStreamName('my/stream')).toBe(false);
    });

    test('rejects special characters', () => {
        expect(isValidStreamName('stream@name!')).toBe(false);
    });
});

// ── isValidInterval ──

describe('isValidInterval', () => {
    test('accepts minimum value 10', () => {
        expect(isValidInterval(10)).toBe(true);
    });

    test('accepts maximum value 86400', () => {
        expect(isValidInterval(86400)).toBe(true);
    });

    test('accepts value in range', () => {
        expect(isValidInterval(300)).toBe(true);
    });

    test('rejects 9 (below minimum)', () => {
        expect(isValidInterval(9)).toBe(false);
    });

    test('rejects 86401 (above maximum)', () => {
        expect(isValidInterval(86401)).toBe(false);
    });

    test('rejects 0', () => {
        expect(isValidInterval(0)).toBe(false);
    });

    test('rejects negative number', () => {
        expect(isValidInterval(-60)).toBe(false);
    });

    test('rejects NaN', () => {
        expect(isValidInterval(NaN)).toBe(false);
    });
});

// ── isNameUnique ──

describe('isNameUnique', () => {
    test('returns true when name is unique', () => {
        expect(isNameUnique('newStream', ['alpha', 'beta'])).toBe(true);
    });

    test('returns false when name already exists', () => {
        expect(isNameUnique('alpha', ['alpha', 'beta'])).toBe(false);
    });

    test('is case-sensitive', () => {
        expect(isNameUnique('Alpha', ['alpha', 'beta'])).toBe(true);
    });

    test('returns true for empty existing list', () => {
        expect(isNameUnique('anything', [])).toBe(true);
    });
});

// ── getCrossFieldWarnings ──

describe('getCrossFieldWarnings', () => {
    const makeState = (overrides: Record<string, any> = {}) => ({
        httpMethod: 'GET',
        streams: [
            {
                retriever: {
                    requester: {},
                    recordSelector: {
                        extractor: { field_path: ['data'] },
                    },
                },
            },
        ],
        ...overrides,
    });

    test('returns no warnings when state is valid', () => {
        const warnings = getCrossFieldWarnings(makeState());
        expect(warnings).toEqual([]);
    });

    test('warns when POST method has no request body', () => {
        const warnings = getCrossFieldWarnings(makeState({ httpMethod: 'POST' }));
        expect(warnings).toEqual(
            expect.arrayContaining([
                expect.objectContaining({
                    field: 'httpMethod',
                    message: expect.stringContaining('no request body'),
                }),
            ])
        );
    });

    test('warns when PUT method has no request body', () => {
        const warnings = getCrossFieldWarnings(
            makeState({
                httpMethod: 'PUT',
                streams: [
                    {
                        retriever: {
                            requester: {},
                            recordSelector: {
                                extractor: { field_path: ['data'] },
                            },
                        },
                    },
                ],
            })
        );
        expect(warnings.some((w: FieldError) => w.field === 'httpMethod')).toBe(true);
    });

    test('warns when PATCH method has no request body', () => {
        const warnings = getCrossFieldWarnings(
            makeState({
                httpMethod: 'PATCH',
                streams: [
                    {
                        retriever: {
                            requester: {},
                            recordSelector: {
                                extractor: { field_path: ['data'] },
                            },
                        },
                    },
                ],
            })
        );
        expect(warnings.some((w: FieldError) => w.field === 'httpMethod')).toBe(true);
    });

    test('no method warning when POST has request_body', () => {
        const state = makeState({
            httpMethod: 'POST',
            streams: [
                {
                    retriever: {
                        requester: { request_body: { type: 'JsonBody', value: '{}' } },
                        recordSelector: {
                            extractor: { field_path: ['data'] },
                        },
                    },
                },
            ],
        });
        const warnings = getCrossFieldWarnings(state);
        expect(warnings.some((w: FieldError) => w.field === 'httpMethod')).toBe(false);
    });

    test('warns when incrementalSync configured but no cursor_field', () => {
        const state = makeState({
            streams: [
                {
                    retriever: {
                        requester: {},
                        recordSelector: {
                            extractor: { field_path: ['data'] },
                        },
                    },
                    incrementalSync: { type: 'DatetimeBasedCursor' },
                },
            ],
        });
        const warnings = getCrossFieldWarnings(state);
        expect(warnings).toEqual(
            expect.arrayContaining([
                expect.objectContaining({
                    field: 'incrementalSync',
                    message: expect.stringContaining('no cursor field'),
                }),
            ])
        );
    });

    test('no incremental warning when cursor_field is set', () => {
        const state = makeState({
            streams: [
                {
                    retriever: {
                        requester: {},
                        recordSelector: {
                            extractor: { field_path: ['data'] },
                        },
                    },
                    incrementalSync: {
                        type: 'DatetimeBasedCursor',
                        cursor_field: 'updated_at',
                    },
                },
            ],
        });
        const warnings = getCrossFieldWarnings(state);
        expect(warnings.some((w: FieldError) => w.field === 'incrementalSync')).toBe(false);
    });

    test('warns when paginator configured but record selector extractors empty', () => {
        const state = makeState({
            streams: [
                {
                    retriever: {
                        requester: {},
                        recordSelector: {
                            extractor: { field_path: [] },
                        },
                        paginator: { type: 'OffsetIncrement', page_size: 100 },
                    },
                },
            ],
        });
        const warnings = getCrossFieldWarnings(state);
        expect(warnings).toEqual(
            expect.arrayContaining([
                expect.objectContaining({
                    field: 'paginator',
                    message: expect.stringContaining('no record selector path'),
                }),
            ])
        );
    });

    test('no paginator warning when record selector has extractors', () => {
        const state = makeState({
            streams: [
                {
                    retriever: {
                        requester: {},
                        recordSelector: {
                            extractor: { field_path: ['results'] },
                        },
                        paginator: { type: 'OffsetIncrement', page_size: 100 },
                    },
                },
            ],
        });
        const warnings = getCrossFieldWarnings(state);
        expect(warnings.some((w: FieldError) => w.field === 'paginator')).toBe(false);
    });

    test('returns multiple warnings when multiple issues exist', () => {
        const state = {
            httpMethod: 'POST',
            streams: [
                {
                    retriever: {
                        requester: {},
                        recordSelector: {
                            extractor: { field_path: [] },
                        },
                        paginator: { type: 'OffsetIncrement' },
                    },
                    incrementalSync: { type: 'DatetimeBasedCursor' },
                },
            ],
        };
        const warnings = getCrossFieldWarnings(state);
        expect(warnings.length).toBe(3);
    });
});
