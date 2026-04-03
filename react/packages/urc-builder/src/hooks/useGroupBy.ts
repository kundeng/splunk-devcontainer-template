import { useState, useMemo } from 'react';
import { parseTags } from '../utils/tag-utils';

export interface Group<T> {
    label: string;
    items: T[];
}

/**
 * Hook for grouping items by a field value. Manages the active group key
 * and returns a memoized array of labelled groups.
 *
 * Special behaviour for `groupKey === 'tags'`: comma-separated tags are
 * parsed so that each item appears in every tag group it belongs to.
 *
 * Items with an empty/missing value for the group key are placed into an
 * "Ungrouped" bucket that always sorts last.
 */
export function useGroupBy<T extends Record<string, any>>(
    items: T[],
    initialGroupKey: string | null = null
): {
    groups: Group<T>[] | null;
    groupKey: string | null;
    setGroupKey: (key: string | null) => void;
} {
    const [groupKey, setGroupKey] = useState<string | null>(initialGroupKey);

    const groups = useMemo<Group<T>[] | null>(() => {
        if (groupKey === null) {
            return null;
        }

        const map = new Map<string, T[]>();

        for (const item of items) {
            const raw = item[groupKey];
            const value = raw == null ? '' : String(raw);

            if (groupKey === 'tags' && value !== '') {
                const tags = parseTags(value);
                if (tags.length === 0) {
                    const bucket = map.get('Ungrouped') ?? [];
                    bucket.push(item);
                    map.set('Ungrouped', bucket);
                } else {
                    for (const tag of tags) {
                        const bucket = map.get(tag) ?? [];
                        bucket.push(item);
                        map.set(tag, bucket);
                    }
                }
            } else if (value === '') {
                const bucket = map.get('Ungrouped') ?? [];
                bucket.push(item);
                map.set('Ungrouped', bucket);
            } else {
                const bucket = map.get(value) ?? [];
                bucket.push(item);
                map.set(value, bucket);
            }
        }

        const result: Group<T>[] = [];
        let ungrouped: Group<T> | null = null;

        for (const [label, groupItems] of map) {
            if (label === 'Ungrouped') {
                ungrouped = { label, items: groupItems };
            } else {
                result.push({ label, items: groupItems });
            }
        }

        result.sort((a, b) => a.label.localeCompare(b.label));

        if (ungrouped) {
            result.push(ungrouped);
        }

        return result;
    }, [items, groupKey]);

    return { groups, groupKey, setGroupKey };
}
