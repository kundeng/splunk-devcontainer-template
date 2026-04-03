import { useState, useMemo } from 'react';

interface SortState {
    key: string | null;
    direction: 'asc' | 'desc';
}

/**
 * Hook for sortable column headers. Manages sort key, direction toggling,
 * and returns a memoized sorted copy of the input array.
 */
export function useSort<T extends Record<string, any>>(
    items: T[],
    defaultKey: string | null = null
): {
    sortedItems: T[];
    sortKey: string | null;
    sortDir: 'asc' | 'desc';
    onSort: (key: string) => void;
} {
    const [sort, setSort] = useState<SortState>({
        key: defaultKey,
        direction: 'asc',
    });

    const onSort = (key: string) => {
        setSort((prev) => {
            if (prev.key === key) {
                return { key, direction: prev.direction === 'asc' ? 'desc' : 'asc' };
            }
            return { key, direction: 'asc' };
        });
    };

    const sortedItems = useMemo(() => {
        if (sort.key === null) {
            return items;
        }

        const key = sort.key;
        const dir = sort.direction === 'asc' ? 1 : -1;

        return [...items].sort((a, b) => {
            const aVal = a[key];
            const bVal = b[key];

            // Null/undefined sort to end regardless of direction
            const aNull = aVal == null;
            const bNull = bVal == null;
            if (aNull && bNull) return 0;
            if (aNull) return 1;
            if (bNull) return -1;

            // Detect numeric values and compare numerically
            const aNum = Number(aVal);
            const bNum = Number(bVal);
            if (!Number.isNaN(aNum) && !Number.isNaN(bNum)) {
                return (aNum - bNum) * dir;
            }

            // General string comparison
            return String(aVal).localeCompare(String(bVal)) * dir;
        });
    }, [items, sort.key, sort.direction]);

    return { sortedItems, sortKey: sort.key, sortDir: sort.direction, onSort };
}
