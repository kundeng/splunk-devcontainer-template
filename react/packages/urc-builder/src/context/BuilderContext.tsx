/**
 * BuilderContext — React Context + useReducer for the stream builder form.
 *
 * Manages all form state across the 5 builder tabs.
 * Supports nested dot-path updates via SET_FIELD action.
 */

import React, { createContext, useContext, useReducer, useMemo } from 'react';
import type { BuilderState, BuilderAction, SavedInput } from '../types';
import { INITIAL_BUILDER_STATE, manifestToFormState } from '../services/manifest-serializer';

// ── Dot-path helper ──

/**
 * Set a value at a nested dot-path in an object.
 * Supports array notation: "streams[0].retriever.type"
 */
function setAtPath(obj: any, path: string, value: any): any {
    const keys = path.replace(/\[(\d+)\]/g, '.$1').split('.');
    const result = Array.isArray(obj) ? [...obj] : { ...obj };
    let current: any = result;

    for (let i = 0; i < keys.length - 1; i++) {
        const key = keys[i];
        const nextKey = keys[i + 1];
        const isNextArray = /^\d+$/.test(nextKey);

        if (/^\d+$/.test(key)) {
            const idx = parseInt(key, 10);
            current[idx] = Array.isArray(current[idx])
                ? [...current[idx]]
                : isNextArray
                  ? [...(current[idx] || [])]
                  : { ...(current[idx] || {}) };
            current = current[idx];
        } else {
            current[key] = Array.isArray(current[key])
                ? [...current[key]]
                : isNextArray
                  ? [...(current[key] || [])]
                  : { ...(current[key] || {}) };
            current = current[key];
        }
    }

    const lastKey = keys[keys.length - 1];
    if (/^\d+$/.test(lastKey)) {
        current[parseInt(lastKey, 10)] = value;
    } else {
        current[lastKey] = value;
    }

    return result;
}

// ── Reducer ──

function builderReducer(state: BuilderState, action: BuilderAction): BuilderState {
    switch (action.type) {
        case 'SET_FIELD':
            return { ...setAtPath(state, action.path, action.value), isDirty: true };

        case 'LOAD_INPUT': {
            const input = action.input;
            const parsed = manifestToFormState(input.manifest, {
                mode: 'edit',
                inputName: input.name,
                account: input.account,
                baseUrl: input.baseUrl,
                index: input.index,
                sourcetype: input.sourcetype,
                interval: input.interval,
                tags: input.tags,
                enabled: !input.disabled,
                debug: input.debug,
            });
            return {
                ...INITIAL_BUILDER_STATE,
                ...parsed,
                mode: 'edit',
                inputName: input.name,
                isDirty: false,
            };
        }

        case 'SET_TEST_RESULT':
            return {
                ...state,
                testResult: action.result,
                testLoading: false,
                validationResults: action.result.validationResults || state.validationResults,
            };

        case 'SET_VALIDATION':
            return { ...state, validationResults: action.results };

        case 'SET_TEST_LOADING':
            return { ...state, testLoading: action.loading };

        case 'RESET':
            return { ...INITIAL_BUILDER_STATE };

        default:
            return state;
    }
}

// ── Context ──

interface BuilderContextValue {
    state: BuilderState;
    dispatch: React.Dispatch<BuilderAction>;
}

const BuilderContext = createContext<BuilderContextValue | null>(null);

export function BuilderProvider({ children }: { children: React.ReactNode }) {
    const [state, dispatch] = useReducer(builderReducer, INITIAL_BUILDER_STATE);
    const value = useMemo(() => ({ state, dispatch }), [state]);

    return (
        <BuilderContext.Provider value={value}>
            {children}
        </BuilderContext.Provider>
    );
}

export function useBuilder(): BuilderContextValue {
    const ctx = useContext(BuilderContext);
    if (!ctx) {
        throw new Error('useBuilder must be used within a BuilderProvider');
    }
    return ctx;
}
