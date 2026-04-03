import { useState, useCallback } from 'react';
import { isValidUrl, isValidStreamName, isValidInterval, isNameUnique } from '../utils/validators';

/**
 * Per-field validation with onBlur integration.
 * Returns error messages keyed by field path, and helpers to validate/clear.
 */
export function useFieldValidation(existingNames: string[]): {
    errors: Record<string, string>;
    validateField: (path: string, value: any) => string | null;
    clearError: (path: string) => void;
} {
    const [errors, setErrors] = useState<Record<string, string>>({});

    const validateField = useCallback(
        (path: string, value: any): string | null => {
            let error: string | null = null;

            switch (path) {
                case 'inputName':
                case 'streams[0].name': {
                    const name = String(value || '');
                    if (!name) {
                        error = 'Stream name is required.';
                    } else if (!isValidStreamName(name)) {
                        error = 'Only letters, numbers, underscores, and hyphens are allowed.';
                    } else if (!isNameUnique(name, existingNames)) {
                        error = 'A stream with this name already exists.';
                    }
                    break;
                }
                case 'baseUrl': {
                    const url = String(value || '');
                    if (!url) {
                        error = 'Base URL is required.';
                    } else if (!isValidUrl(url)) {
                        error = 'Base URL must start with http:// or https://';
                    }
                    break;
                }
                case 'interval': {
                    const num = Number(value);
                    if (!isValidInterval(num)) {
                        error = 'Interval must be between 10 and 86,400 seconds.';
                    }
                    break;
                }
                default:
                    break;
            }

            setErrors((prev) => {
                if (error) {
                    return { ...prev, [path]: error };
                }
                const next = { ...prev };
                delete next[path];
                return next;
            });

            return error;
        },
        [existingNames]
    );

    const clearError = useCallback((path: string) => {
        setErrors((prev) => {
            if (!(path in prev)) return prev;
            const next = { ...prev };
            delete next[path];
            return next;
        });
    }, []);

    return { errors, validateField, clearError };
}
