import { renderHook, act } from '@testing-library/react';
import { useGroupBy } from './useGroupBy';

const items = [
    { name: 'stream-1', account: 'github-acct', tags: 'devops, security' },
    { name: 'stream-2', account: 'slack-acct', tags: 'devops' },
    { name: 'stream-3', account: 'github-acct', tags: '' },
    { name: 'stream-4', account: '', tags: 'security' },
];

describe('useGroupBy', () => {
    it('returns null groups when groupKey is null', () => {
        const { result } = renderHook(() => useGroupBy(items));
        expect(result.current.groups).toBeNull();
        expect(result.current.groupKey).toBeNull();
    });

    it('groups by a regular string field (account)', () => {
        const { result } = renderHook(() => useGroupBy(items, 'account'));
        const { groups } = result.current;

        expect(groups).not.toBeNull();
        expect(groups).toHaveLength(3);

        const github = groups!.find((g) => g.label === 'github-acct');
        const slack = groups!.find((g) => g.label === 'slack-acct');
        const ungrouped = groups!.find((g) => g.label === 'Ungrouped');

        expect(github!.items).toHaveLength(2);
        expect(slack!.items).toHaveLength(1);
        expect(ungrouped!.items).toHaveLength(1);
    });

    it('groups by tags with multi-group membership', () => {
        const { result } = renderHook(() => useGroupBy(items, 'tags'));
        const { groups } = result.current;

        expect(groups).not.toBeNull();
        expect(groups).toHaveLength(3);

        const devops = groups!.find((g) => g.label === 'devops');
        const security = groups!.find((g) => g.label === 'security');
        const ungrouped = groups!.find((g) => g.label === 'Ungrouped');

        expect(devops!.items).toHaveLength(2);
        expect(security!.items).toHaveLength(2);
        expect(ungrouped!.items).toHaveLength(1);
    });

    it('sorts groups alphabetically with Ungrouped last', () => {
        const { result } = renderHook(() => useGroupBy(items, 'account'));
        const labels = result.current.groups!.map((g) => g.label);

        expect(labels).toEqual(['github-acct', 'slack-acct', 'Ungrouped']);
    });

    it('sorts tag groups alphabetically with Ungrouped last', () => {
        const { result } = renderHook(() => useGroupBy(items, 'tags'));
        const labels = result.current.groups!.map((g) => g.label);

        expect(labels).toEqual(['devops', 'security', 'Ungrouped']);
    });

    it('changes grouping via setGroupKey', () => {
        const { result } = renderHook(() => useGroupBy(items));

        expect(result.current.groups).toBeNull();

        act(() => {
            result.current.setGroupKey('account');
        });

        expect(result.current.groupKey).toBe('account');
        expect(result.current.groups).toHaveLength(3);
    });

    it('returns empty array (not null) when items is empty and groupKey is set', () => {
        const { result } = renderHook(() => useGroupBy([], 'account'));

        expect(result.current.groups).not.toBeNull();
        expect(result.current.groups).toEqual([]);
    });

    it('returns null groups when setGroupKey is called with null', () => {
        const { result } = renderHook(() => useGroupBy(items, 'account'));

        expect(result.current.groups).not.toBeNull();

        act(() => {
            result.current.setGroupKey(null);
        });

        expect(result.current.groups).toBeNull();
        expect(result.current.groupKey).toBeNull();
    });
});
