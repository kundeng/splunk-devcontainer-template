import { useState, useMemo, useEffect, useRef } from 'react';

/**
 * Hook for search/text filtering. Manages a search query with debounced
 * filtering and returns items that match the query across the specified keys.
 */
export function useSearch<T extends Record<string, any>>(
    items: T[],
    keys: (keyof T)[],
    debounceMs: number = 300
): {
    query: string;
    setQuery: (q: string) => void;
    searchedItems: T[];
} {
    const [query, setQuery] = useState('');
    const [debouncedQuery, setDebouncedQuery] = useState('');
    const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    useEffect(() => {
        if (timerRef.current !== null) {
            clearTimeout(timerRef.current);
        }
        timerRef.current = setTimeout(() => {
            setDebouncedQuery(query);
            timerRef.current = null;
        }, debounceMs);

        return () => {
            if (timerRef.current !== null) {
                clearTimeout(timerRef.current);
            }
        };
    }, [query, debounceMs]);

    const searchedItems = useMemo(() => {
        const trimmed = debouncedQuery.trim().toLowerCase();
        if (trimmed === '') {
            return items;
        }

        return items.filter((item) =>
            keys.some((key) => {
                const val = item[key];
                if (val == null) return false;
                return String(val).toLowerCase().includes(trimmed);
            })
        );
    }, [items, keys, debouncedQuery]);

    return { query, setQuery, searchedItems };
}
