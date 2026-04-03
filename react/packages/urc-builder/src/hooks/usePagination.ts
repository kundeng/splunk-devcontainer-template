import { useState, useMemo, useEffect } from 'react';

export function usePagination<T>(
    items: T[],
    defaultPageSize: number = 25
): {
    pagedItems: T[];
    page: number;
    pageSize: number;
    totalPages: number;
    setPage: (n: number) => void;
    setPageSize: (n: number) => void;
} {
    const [page, setPageRaw] = useState(0);
    const [pageSize, setPageSizeRaw] = useState(defaultPageSize);

    const totalPages = useMemo(
        () => Math.max(1, Math.ceil(items.length / pageSize)),
        [items.length, pageSize]
    );

    // Reset to page 0 when items array reference changes
    useEffect(() => {
        setPageRaw(0);
    }, [items]);

    const setPage = (n: number) => {
        const clamped = Math.max(0, Math.min(n, totalPages - 1));
        setPageRaw(clamped);
    };

    const setPageSize = (n: number) => {
        setPageSizeRaw(n);
        setPageRaw(0);
    };

    const pagedItems = useMemo(
        () => items.slice(page * pageSize, (page + 1) * pageSize),
        [items, page, pageSize]
    );

    return { pagedItems, page, pageSize, totalPages, setPage, setPageSize };
}
