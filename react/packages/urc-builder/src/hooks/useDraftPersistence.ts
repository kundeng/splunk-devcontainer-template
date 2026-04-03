import { useEffect, useCallback, useState } from 'react';
import type { BuilderState } from '../types';

const STORAGE_PREFIX = 'urc-draft-';

// Fields to exclude from draft (ephemeral test state)
const EXCLUDED_KEYS: (keyof BuilderState)[] = ['testResult', 'testLoading', 'validationResults'];

function getDraftKey(mode: string, inputName: string | null): string {
    return `${STORAGE_PREFIX}${mode}-${inputName || 'new'}`;
}

function sanitizeForStorage(state: BuilderState): Partial<BuilderState> {
    const copy: any = { ...state };
    for (const key of EXCLUDED_KEYS) {
        delete copy[key];
    }
    return copy;
}

export function useDraftPersistence(
    state: BuilderState,
    mode: string,
    inputName: string | null
): {
    hasDraft: boolean;
    restoreDraft: () => Partial<BuilderState> | null;
    discardDraft: () => void;
    clearDraft: () => void;
} {
    const key = getDraftKey(mode, inputName);
    const [hasDraft, setHasDraft] = useState(() => {
        try {
            return localStorage.getItem(key) !== null;
        } catch {
            return false;
        }
    });

    // Auto-save with 2s debounce
    useEffect(() => {
        if (!state.isDirty) return;

        const timer = setTimeout(() => {
            try {
                localStorage.setItem(key, JSON.stringify(sanitizeForStorage(state)));
            } catch {
                // QuotaExceededError — silently skip
            }
        }, 2000);

        return () => clearTimeout(timer);
    }, [state, key]);

    const restoreDraft = useCallback((): Partial<BuilderState> | null => {
        try {
            const raw = localStorage.getItem(key);
            if (raw) return JSON.parse(raw);
        } catch {
            // corrupt data
        }
        return null;
    }, [key]);

    const discardDraft = useCallback(() => {
        try {
            localStorage.removeItem(key);
        } catch {
            // ignore
        }
        setHasDraft(false);
    }, [key]);

    const clearDraft = useCallback(() => {
        try {
            localStorage.removeItem(key);
        } catch {
            // ignore
        }
        setHasDraft(false);
    }, [key]);

    return { hasDraft, restoreDraft, discardDraft, clearDraft };
}
