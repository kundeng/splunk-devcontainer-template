import { renderHook, act } from '@testing-library/react';
import { useSearch } from './useSearch';

const items = [
    { name: 'github-events', baseUrl: 'https://api.github.com' },
    { name: 'slack-messages', baseUrl: 'https://slack.com/api' },
    { name: 'jira-issues', baseUrl: 'https://jira.atlassian.net' },
];

beforeEach(() => {
    jest.useFakeTimers();
});

afterEach(() => {
    jest.useRealTimers();
});

describe('useSearch', () => {
    it('returns all items for empty query', () => {
        const { result } = renderHook(() =>
            useSearch(items, ['name', 'baseUrl'])
        );
        expect(result.current.searchedItems).toEqual(items);
        expect(result.current.query).toBe('');
    });

    it('filters by name after debounce', () => {
        const { result } = renderHook(() =>
            useSearch(items, ['name', 'baseUrl'])
        );

        act(() => {
            result.current.setQuery('github');
        });

        // Advance past debounce
        act(() => {
            jest.advanceTimersByTime(300);
        });

        expect(result.current.searchedItems).toEqual([
            { name: 'github-events', baseUrl: 'https://api.github.com' },
        ]);
    });

    it('filters by baseUrl after debounce', () => {
        const { result } = renderHook(() =>
            useSearch(items, ['name', 'baseUrl'])
        );

        act(() => {
            result.current.setQuery('slack.com');
        });
        act(() => {
            jest.advanceTimersByTime(300);
        });

        expect(result.current.searchedItems).toEqual([
            { name: 'slack-messages', baseUrl: 'https://slack.com/api' },
        ]);
    });

    it('matches case-insensitively', () => {
        const { result } = renderHook(() =>
            useSearch(items, ['name', 'baseUrl'])
        );

        act(() => {
            result.current.setQuery('GITHUB');
        });
        act(() => {
            jest.advanceTimersByTime(300);
        });

        expect(result.current.searchedItems).toEqual([
            { name: 'github-events', baseUrl: 'https://api.github.com' },
        ]);
    });

    it('returns empty array when nothing matches', () => {
        const { result } = renderHook(() =>
            useSearch(items, ['name', 'baseUrl'])
        );

        act(() => {
            result.current.setQuery('zzz-no-match');
        });
        act(() => {
            jest.advanceTimersByTime(300);
        });

        expect(result.current.searchedItems).toEqual([]);
    });

    it('matches across multiple keys', () => {
        const { result } = renderHook(() =>
            useSearch(items, ['name', 'baseUrl'])
        );

        // "atlassian" appears in jira's baseUrl only
        act(() => {
            result.current.setQuery('atlassian');
        });
        act(() => {
            jest.advanceTimersByTime(300);
        });

        expect(result.current.searchedItems).toEqual([
            { name: 'jira-issues', baseUrl: 'https://jira.atlassian.net' },
        ]);

        // "api" appears in github's baseUrl and slack's baseUrl
        act(() => {
            result.current.setQuery('api');
        });
        act(() => {
            jest.advanceTimersByTime(300);
        });

        expect(result.current.searchedItems).toEqual([
            { name: 'github-events', baseUrl: 'https://api.github.com' },
            { name: 'slack-messages', baseUrl: 'https://slack.com/api' },
        ]);
    });

    it('does not filter before debounce expires', () => {
        const { result } = renderHook(() =>
            useSearch(items, ['name', 'baseUrl'])
        );

        act(() => {
            result.current.setQuery('github');
        });

        // Before debounce: query is updated but items are NOT filtered yet
        expect(result.current.query).toBe('github');
        expect(result.current.searchedItems).toEqual(items);

        // After debounce: items are filtered
        act(() => {
            jest.advanceTimersByTime(300);
        });

        expect(result.current.searchedItems).toEqual([
            { name: 'github-events', baseUrl: 'https://api.github.com' },
        ]);
    });

    it('restores all items when query is cleared', () => {
        const { result } = renderHook(() =>
            useSearch(items, ['name', 'baseUrl'])
        );

        // Set a filter
        act(() => {
            result.current.setQuery('github');
        });
        act(() => {
            jest.advanceTimersByTime(300);
        });

        expect(result.current.searchedItems).toHaveLength(1);

        // Clear the filter
        act(() => {
            result.current.setQuery('');
        });
        act(() => {
            jest.advanceTimersByTime(300);
        });

        expect(result.current.searchedItems).toEqual(items);
    });
});
