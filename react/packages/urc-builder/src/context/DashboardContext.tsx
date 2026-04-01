/**
 * DashboardContext — React Context for the Stream Dashboard filters and state.
 *
 * Lighter than BuilderContext — uses useState, not useReducer.
 */

import React, { createContext, useContext, useState, useMemo, useCallback } from 'react';
import type { InputSummary, DashboardState } from '../types';

interface DashboardContextValue {
    state: DashboardState;
    setInputs: (inputs: InputSummary[]) => void;
    setSelectedTags: (tags: string[]) => void;
    setSelectedStatus: (status: DashboardState['selectedStatus']) => void;
    setSelectedRows: (rows: string[]) => void;
    setLoading: (loading: boolean) => void;
    setError: (error: string | null) => void;
    filteredInputs: InputSummary[];
    allTags: string[];
}

const DashboardContext = createContext<DashboardContextValue | null>(null);

const INITIAL_STATE: DashboardState = {
    inputs: [],
    selectedTags: [],
    selectedStatus: 'all',
    selectedRows: [],
    loading: true,
    error: null,
};

export function DashboardProvider({ children }: { children: React.ReactNode }) {
    const [state, setState] = useState<DashboardState>(INITIAL_STATE);

    const setInputs = useCallback((inputs: InputSummary[]) => {
        setState((s) => ({ ...s, inputs, loading: false, error: null }));
    }, []);

    const setSelectedTags = useCallback((tags: string[]) => {
        setState((s) => ({ ...s, selectedTags: tags }));
    }, []);

    const setSelectedStatus = useCallback((status: DashboardState['selectedStatus']) => {
        setState((s) => ({ ...s, selectedStatus: status }));
    }, []);

    const setSelectedRows = useCallback((rows: string[]) => {
        setState((s) => ({ ...s, selectedRows: rows }));
    }, []);

    const setLoading = useCallback((loading: boolean) => {
        setState((s) => ({ ...s, loading }));
    }, []);

    const setError = useCallback((error: string | null) => {
        setState((s) => ({ ...s, error, loading: false }));
    }, []);

    // Derived: unique tags from all inputs
    const allTags = useMemo(() => {
        const tagSet = new Set<string>();
        for (const input of state.inputs) {
            if (input.tags) {
                input.tags.split(',').forEach((t) => {
                    const trimmed = t.trim();
                    if (trimmed) tagSet.add(trimmed);
                });
            }
        }
        return Array.from(tagSet).sort();
    }, [state.inputs]);

    // Derived: filtered inputs
    const filteredInputs = useMemo(() => {
        let result = state.inputs;

        // Filter by tags
        if (state.selectedTags.length > 0) {
            result = result.filter((input) => {
                const inputTags = (input.tags || '').split(',').map((t) => t.trim());
                return state.selectedTags.some((tag) => inputTags.includes(tag));
            });
        }

        // Filter by status
        if (state.selectedStatus !== 'all') {
            result = result.filter((input) => {
                if (state.selectedStatus === 'active') return !input.disabled;
                if (state.selectedStatus === 'disabled') return input.disabled;
                // 'error' status — we don't have runtime error state from UCC REST yet,
                // so this filters nothing for now. Will be enhanced when we add
                // runtime status from structured logs.
                return true;
            });
        }

        return result;
    }, [state.inputs, state.selectedTags, state.selectedStatus]);

    const value = useMemo(
        () => ({
            state,
            setInputs,
            setSelectedTags,
            setSelectedStatus,
            setSelectedRows,
            setLoading,
            setError,
            filteredInputs,
            allTags,
        }),
        [state, filteredInputs, allTags, setInputs, setSelectedTags, setSelectedStatus, setSelectedRows, setLoading, setError]
    );

    return (
        <DashboardContext.Provider value={value}>
            {children}
        </DashboardContext.Provider>
    );
}

export function useDashboard(): DashboardContextValue {
    const ctx = useContext(DashboardContext);
    if (!ctx) {
        throw new Error('useDashboard must be used within a DashboardProvider');
    }
    return ctx;
}
