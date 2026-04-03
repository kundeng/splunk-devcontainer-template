import { renderHook, act } from '@testing-library/react';
import { usePagination } from './usePagination';

function makeItems(n: number): number[] {
    return Array.from({ length: n }, (_, i) => i);
}

describe('usePagination', () => {
    test('defaults to page 0 and pageSize 25', () => {
        const { result } = renderHook(() => usePagination(makeItems(50)));
        expect(result.current.page).toBe(0);
        expect(result.current.pageSize).toBe(25);
    });

    test('returns correct slice for page 0 with 30 items', () => {
        const items = makeItems(30);
        const { result } = renderHook(() => usePagination(items));
        expect(result.current.pagedItems).toHaveLength(25);
        expect(result.current.pagedItems).toEqual(items.slice(0, 25));
    });

    test('returns correct slice for page 1 with 30 items', () => {
        const items = makeItems(30);
        const { result } = renderHook(() => usePagination(items));

        act(() => {
            result.current.setPage(1);
        });

        expect(result.current.pagedItems).toHaveLength(5);
        expect(result.current.pagedItems).toEqual(items.slice(25, 30));
    });

    test('setPageSize changes size and resets to page 0', () => {
        const items = makeItems(50);
        const { result } = renderHook(() => usePagination(items));

        // Navigate to page 1 first
        act(() => {
            result.current.setPage(1);
        });
        expect(result.current.page).toBe(1);

        // Change page size
        act(() => {
            result.current.setPageSize(10);
        });
        expect(result.current.pageSize).toBe(10);
        expect(result.current.page).toBe(0);
        expect(result.current.pagedItems).toHaveLength(10);
    });

    test('items change resets to page 0', () => {
        let items = makeItems(50);
        const { result, rerender } = renderHook(
            ({ items: hookItems }) => usePagination(hookItems),
            { initialProps: { items } }
        );

        act(() => {
            result.current.setPage(1);
        });
        expect(result.current.page).toBe(1);

        // Provide a new array reference
        items = makeItems(60);
        rerender({ items });

        expect(result.current.page).toBe(0);
    });

    test('empty array returns empty pagedItems, totalPages 1, page 0', () => {
        const { result } = renderHook(() => usePagination<number>([]));
        expect(result.current.pagedItems).toEqual([]);
        expect(result.current.totalPages).toBe(1);
        expect(result.current.page).toBe(0);
    });

    test('out-of-bounds setPage is clamped to valid range', () => {
        const items = makeItems(50);
        const { result } = renderHook(() => usePagination(items));
        // totalPages = 2 (pages 0 and 1)

        act(() => {
            result.current.setPage(100);
        });
        expect(result.current.page).toBe(1);

        act(() => {
            result.current.setPage(-5);
        });
        expect(result.current.page).toBe(0);
    });

    test.each([
        [0, 1],
        [1, 1],
        [25, 1],
        [26, 2],
        [50, 2],
        [51, 3],
    ])('totalPages for %i items is %i', (count, expected) => {
        const { result } = renderHook(() => usePagination(makeItems(count)));
        expect(result.current.totalPages).toBe(expected);
    });

    test('custom defaultPageSize', () => {
        const items = makeItems(25);
        const { result } = renderHook(() => usePagination(items, 10));
        expect(result.current.pageSize).toBe(10);
        expect(result.current.pagedItems).toHaveLength(10);
        expect(result.current.totalPages).toBe(3);
    });
});
