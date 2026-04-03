import { renderHook, act } from '@testing-library/react';
import { useDraftPersistence } from './useDraftPersistence';
import type { BuilderState } from '../types';

// ── localStorage mock ──

const mockStorage: Record<string, string> = {};
beforeEach(() => {
    Object.keys(mockStorage).forEach((k) => delete mockStorage[k]);
    jest.spyOn(Storage.prototype, 'getItem').mockImplementation((k) => mockStorage[k] ?? null);
    jest.spyOn(Storage.prototype, 'setItem').mockImplementation((k, v) => {
        mockStorage[k] = v;
    });
    jest.spyOn(Storage.prototype, 'removeItem').mockImplementation((k) => {
        delete mockStorage[k];
    });
    jest.useFakeTimers();
});
afterEach(() => {
    jest.restoreAllMocks();
    jest.useRealTimers();
});

// ── Helpers ──

function makeState(overrides: Partial<BuilderState> = {}): BuilderState {
    return {
        mode: 'create',
        inputName: null,
        account: '',
        baseUrl: 'https://api.example.com',
        httpMethod: 'GET',
        streams: [],
        index: 'main',
        sourcetype: 'urc',
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
        ...overrides,
    };
}

describe('useDraftPersistence', () => {
    test('save triggers after 2s debounce', () => {
        const state = makeState({ isDirty: true });
        renderHook(() => useDraftPersistence(state, 'create', null));

        // Not yet saved
        expect(mockStorage['urc-draft-create-new']).toBeUndefined();

        // Advance past debounce
        act(() => {
            jest.advanceTimersByTime(2000);
        });

        expect(mockStorage['urc-draft-create-new']).toBeDefined();
    });

    test('restore returns saved state', () => {
        const state = makeState({ isDirty: true, baseUrl: 'https://saved.example.com' });
        const { result, rerender } = renderHook(
            ({ s }) => useDraftPersistence(s, 'create', null),
            { initialProps: { s: state } }
        );

        act(() => {
            jest.advanceTimersByTime(2000);
        });

        // Now restore
        const draft = result.current.restoreDraft();
        expect(draft).not.toBeNull();
        expect((draft as any).baseUrl).toBe('https://saved.example.com');
    });

    test('discard clears localStorage', () => {
        mockStorage['urc-draft-create-new'] = JSON.stringify({ baseUrl: 'x' });
        const state = makeState();
        const { result } = renderHook(() => useDraftPersistence(state, 'create', null));

        act(() => {
            result.current.discardDraft();
        });

        expect(mockStorage['urc-draft-create-new']).toBeUndefined();
        expect(result.current.hasDraft).toBe(false);
    });

    test('clearDraft removes entry', () => {
        mockStorage['urc-draft-edit-myinput'] = JSON.stringify({ baseUrl: 'x' });
        const state = makeState({ mode: 'edit', inputName: 'myinput' });
        const { result } = renderHook(() => useDraftPersistence(state, 'edit', 'myinput'));

        act(() => {
            result.current.clearDraft();
        });

        expect(mockStorage['urc-draft-edit-myinput']).toBeUndefined();
        expect(result.current.hasDraft).toBe(false);
    });

    test('excludes testResult, testLoading, validationResults from saved data', () => {
        const state = makeState({
            isDirty: true,
            testResult: { status: 'success', message: 'ok' },
            testLoading: true,
            validationResults: [{ path: 'x', type: 'error', severity: 'error', message: 'bad' }],
        });
        renderHook(() => useDraftPersistence(state, 'create', null));

        act(() => {
            jest.advanceTimersByTime(2000);
        });

        const saved = JSON.parse(mockStorage['urc-draft-create-new']);
        expect(saved.testResult).toBeUndefined();
        expect(saved.testLoading).toBeUndefined();
        expect(saved.validationResults).toBeUndefined();
    });

    test('hasDraft is true when localStorage has a matching key', () => {
        mockStorage['urc-draft-create-new'] = JSON.stringify({ baseUrl: 'x' });
        const state = makeState();
        const { result } = renderHook(() => useDraftPersistence(state, 'create', null));

        expect(result.current.hasDraft).toBe(true);
    });

    test('hasDraft is false when no key exists', () => {
        const state = makeState();
        const { result } = renderHook(() => useDraftPersistence(state, 'create', null));

        expect(result.current.hasDraft).toBe(false);
    });

    test('silently handles QuotaExceededError', () => {
        jest.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
            throw new DOMException('quota exceeded', 'QuotaExceededError');
        });

        const state = makeState({ isDirty: true });
        // Should not throw
        renderHook(() => useDraftPersistence(state, 'create', null));

        act(() => {
            jest.advanceTimersByTime(2000);
        });

        // No error thrown — test passes if we reach here
        expect(true).toBe(true);
    });
});
