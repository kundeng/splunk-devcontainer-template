import { renderHook, act } from '@testing-library/react';
import { useFieldValidation } from './useFieldValidation';

describe('useFieldValidation', () => {
    const existingNames = ['github-events', 'slack-messages'];

    test('validates stream name: empty', () => {
        const { result } = renderHook(() => useFieldValidation(existingNames));
        act(() => {
            result.current.validateField('streams[0].name', '');
        });
        expect(result.current.errors['streams[0].name']).toBe('Stream name is required.');
    });

    test('validates stream name: invalid characters', () => {
        const { result } = renderHook(() => useFieldValidation(existingNames));
        act(() => {
            result.current.validateField('streams[0].name', 'my stream!');
        });
        expect(result.current.errors['streams[0].name']).toContain('letters, numbers');
    });

    test('validates stream name: duplicate', () => {
        const { result } = renderHook(() => useFieldValidation(existingNames));
        act(() => {
            result.current.validateField('streams[0].name', 'github-events');
        });
        expect(result.current.errors['streams[0].name']).toContain('already exists');
    });

    test('validates stream name: valid', () => {
        const { result } = renderHook(() => useFieldValidation(existingNames));
        act(() => {
            result.current.validateField('streams[0].name', 'new-stream');
        });
        expect(result.current.errors['streams[0].name']).toBeUndefined();
    });

    test('validates base URL: empty', () => {
        const { result } = renderHook(() => useFieldValidation(existingNames));
        act(() => {
            result.current.validateField('baseUrl', '');
        });
        expect(result.current.errors['baseUrl']).toBe('Base URL is required.');
    });

    test('validates base URL: invalid', () => {
        const { result } = renderHook(() => useFieldValidation(existingNames));
        act(() => {
            result.current.validateField('baseUrl', 'not-a-url');
        });
        expect(result.current.errors['baseUrl']).toContain('http://');
    });

    test('validates base URL: valid', () => {
        const { result } = renderHook(() => useFieldValidation(existingNames));
        act(() => {
            result.current.validateField('baseUrl', 'https://api.example.com');
        });
        expect(result.current.errors['baseUrl']).toBeUndefined();
    });

    test('validates interval: too low', () => {
        const { result } = renderHook(() => useFieldValidation(existingNames));
        act(() => {
            result.current.validateField('interval', 5);
        });
        expect(result.current.errors['interval']).toContain('between 10');
    });

    test('validates interval: valid', () => {
        const { result } = renderHook(() => useFieldValidation(existingNames));
        act(() => {
            result.current.validateField('interval', 60);
        });
        expect(result.current.errors['interval']).toBeUndefined();
    });

    test('clearError removes specific error', () => {
        const { result } = renderHook(() => useFieldValidation(existingNames));
        act(() => {
            result.current.validateField('baseUrl', '');
            result.current.validateField('streams[0].name', '');
        });
        expect(Object.keys(result.current.errors)).toHaveLength(2);

        act(() => {
            result.current.clearError('baseUrl');
        });
        expect(result.current.errors['baseUrl']).toBeUndefined();
        expect(result.current.errors['streams[0].name']).toBeDefined();
    });

    test('valid value clears previous error', () => {
        const { result } = renderHook(() => useFieldValidation(existingNames));
        act(() => {
            result.current.validateField('baseUrl', 'bad');
        });
        expect(result.current.errors['baseUrl']).toBeDefined();

        act(() => {
            result.current.validateField('baseUrl', 'https://api.example.com');
        });
        expect(result.current.errors['baseUrl']).toBeUndefined();
    });

    test('inputName path also validates as stream name', () => {
        const { result } = renderHook(() => useFieldValidation(existingNames));
        act(() => {
            result.current.validateField('inputName', 'github-events');
        });
        expect(result.current.errors['inputName']).toContain('already exists');
    });

    test('unknown path returns no error', () => {
        const { result } = renderHook(() => useFieldValidation(existingNames));
        act(() => {
            result.current.validateField('someOtherField', 'value');
        });
        expect(Object.keys(result.current.errors)).toHaveLength(0);
    });
});
