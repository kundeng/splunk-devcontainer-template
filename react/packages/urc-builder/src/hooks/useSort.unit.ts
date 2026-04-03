import { renderHook, act } from '@testing-library/react';
import { useSort } from './useSort';

const FRUITS = [
    { name: 'banana', count: 5 },
    { name: 'apple', count: 12 },
    { name: 'cherry', count: 2 },
];

describe('useSort', () => {
    it('returns items in original order when sortKey is null', () => {
        const { result } = renderHook(() => useSort(FRUITS));
        expect(result.current.sortKey).toBeNull();
        expect(result.current.sortDir).toBe('asc');
        expect(result.current.sortedItems).toEqual(FRUITS);
    });

    it('sorts ascending by string field', () => {
        const { result } = renderHook(() => useSort(FRUITS));

        act(() => result.current.onSort('name'));

        expect(result.current.sortKey).toBe('name');
        expect(result.current.sortDir).toBe('asc');
        expect(result.current.sortedItems.map((i) => i.name)).toEqual([
            'apple',
            'banana',
            'cherry',
        ]);
    });

    it('sorts descending when same key clicked twice', () => {
        const { result } = renderHook(() => useSort(FRUITS));

        act(() => result.current.onSort('name'));
        act(() => result.current.onSort('name'));

        expect(result.current.sortDir).toBe('desc');
        expect(result.current.sortedItems.map((i) => i.name)).toEqual([
            'cherry',
            'banana',
            'apple',
        ]);
    });

    it('toggles back to ascending on third click', () => {
        const { result } = renderHook(() => useSort(FRUITS));

        act(() => result.current.onSort('name'));
        act(() => result.current.onSort('name'));
        act(() => result.current.onSort('name'));

        expect(result.current.sortDir).toBe('asc');
        expect(result.current.sortedItems.map((i) => i.name)).toEqual([
            'apple',
            'banana',
            'cherry',
        ]);
    });

    it('resets to ascending when a new key is selected', () => {
        const { result } = renderHook(() => useSort(FRUITS));

        // Sort by name descending
        act(() => result.current.onSort('name'));
        act(() => result.current.onSort('name'));
        expect(result.current.sortDir).toBe('desc');

        // Switch to count — should reset to ascending
        act(() => result.current.onSort('count'));
        expect(result.current.sortKey).toBe('count');
        expect(result.current.sortDir).toBe('asc');
    });

    it('sorts numbers numerically, not lexically', () => {
        const items = [
            { label: 'a', value: 10 },
            { label: 'b', value: 2 },
            { label: 'c', value: 100 },
        ];
        const { result } = renderHook(() => useSort(items));

        act(() => result.current.onSort('value'));

        expect(result.current.sortedItems.map((i) => i.value)).toEqual([2, 10, 100]);
    });

    it('sorts numeric strings numerically', () => {
        const items = [
            { id: '10', name: 'x' },
            { id: '2', name: 'y' },
            { id: '100', name: 'z' },
        ];
        const { result } = renderHook(() => useSort(items));

        act(() => result.current.onSort('id'));

        expect(result.current.sortedItems.map((i) => i.id)).toEqual(['2', '10', '100']);
    });

    it('returns empty array for empty input', () => {
        const { result } = renderHook(() => useSort([]));

        act(() => result.current.onSort('anything'));

        expect(result.current.sortedItems).toEqual([]);
    });

    it('sorts null and undefined values to end', () => {
        const items = [
            { name: null, rank: 1 },
            { name: 'alpha', rank: 2 },
            { name: undefined, rank: 3 },
            { name: 'beta', rank: 4 },
        ] as Array<{ name: string | null | undefined; rank: number }>;

        const { result } = renderHook(() => useSort(items));

        act(() => result.current.onSort('name'));

        const names = result.current.sortedItems.map((i) => i.name);
        // Non-null values sorted first, nullish values at the end
        expect(names).toEqual(['alpha', 'beta', null, undefined]);
    });

    it('keeps null/undefined at end even in descending order', () => {
        const items = [
            { name: null, rank: 1 },
            { name: 'alpha', rank: 2 },
            { name: undefined, rank: 3 },
            { name: 'beta', rank: 4 },
        ] as Array<{ name: string | null | undefined; rank: number }>;

        const { result } = renderHook(() => useSort(items));

        act(() => result.current.onSort('name'));
        act(() => result.current.onSort('name'));

        const names = result.current.sortedItems.map((i) => i.name);
        // Descending non-null first, then nullish at end
        expect(names).toEqual(['beta', 'alpha', null, undefined]);
    });

    it('accepts a defaultKey and sorts immediately', () => {
        const { result } = renderHook(() => useSort(FRUITS, 'name'));

        expect(result.current.sortKey).toBe('name');
        expect(result.current.sortDir).toBe('asc');
        expect(result.current.sortedItems.map((i) => i.name)).toEqual([
            'apple',
            'banana',
            'cherry',
        ]);
    });
});
