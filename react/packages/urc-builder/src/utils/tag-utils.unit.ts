import { parseTags, formatTags, getAutocompleteSuggestions } from './tag-utils';

/* ------------------------------------------------------------------ */
/*  parseTags                                                         */
/* ------------------------------------------------------------------ */
describe('parseTags', () => {
    it('returns empty array for empty string', () => {
        expect(parseTags('')).toEqual([]);
    });

    it('parses a single tag', () => {
        expect(parseTags('devops')).toEqual(['devops']);
    });

    it('parses multiple comma-separated tags', () => {
        expect(parseTags('devops,security,prod')).toEqual(['devops', 'security', 'prod']);
    });

    it('trims whitespace around tags', () => {
        expect(parseTags('  devops , security , prod  ')).toEqual(['devops', 'security', 'prod']);
    });

    it('removes duplicate tags (keeps first occurrence)', () => {
        expect(parseTags('devops, security, devops')).toEqual(['devops', 'security']);
    });

    it('handles trailing comma', () => {
        expect(parseTags('devops, security,')).toEqual(['devops', 'security']);
    });

    it('handles leading comma', () => {
        expect(parseTags(',devops, security')).toEqual(['devops', 'security']);
    });

    it('handles multiple consecutive commas', () => {
        expect(parseTags('devops,,security,,,prod')).toEqual(['devops', 'security', 'prod']);
    });

    it('trims a single padded tag', () => {
        expect(parseTags('  single  ')).toEqual(['single']);
    });
});

/* ------------------------------------------------------------------ */
/*  formatTags                                                        */
/* ------------------------------------------------------------------ */
describe('formatTags', () => {
    it('returns empty string for empty array', () => {
        expect(formatTags([])).toBe('');
    });

    it('formats a single tag', () => {
        expect(formatTags(['devops'])).toBe('devops');
    });

    it('joins multiple tags with ", "', () => {
        expect(formatTags(['devops', 'security', 'prod'])).toBe('devops, security, prod');
    });
});

/* ------------------------------------------------------------------ */
/*  Round-trip                                                        */
/* ------------------------------------------------------------------ */
describe('round-trip: formatTags(parseTags(s))', () => {
    it('preserves semantics for well-formed input', () => {
        const input = 'devops, security, prod';
        expect(formatTags(parseTags(input))).toBe(input);
    });

    it('normalises whitespace and removes duplicates', () => {
        const input = ' devops ,security,  devops , prod ';
        expect(formatTags(parseTags(input))).toBe('devops, security, prod');
    });

    it('handles empty string', () => {
        expect(formatTags(parseTags(''))).toBe('');
    });
});

/* ------------------------------------------------------------------ */
/*  getAutocompleteSuggestions                                        */
/* ------------------------------------------------------------------ */
describe('getAutocompleteSuggestions', () => {
    const allTags = ['devops', 'dev-tools', 'security', 'prod', 'DEV-ops'];

    it('returns tags matching the prefix', () => {
        expect(getAutocompleteSuggestions('dev', allTags)).toEqual([
            'devops',
            'dev-tools',
            'DEV-ops',
        ]);
    });

    it('matches case-insensitively', () => {
        expect(getAutocompleteSuggestions('DEV', allTags)).toEqual([
            'devops',
            'dev-tools',
            'DEV-ops',
        ]);
    });

    it('excludes already-selected tags', () => {
        expect(getAutocompleteSuggestions('dev', allTags, ['devops'])).toEqual([
            'dev-tools',
            'DEV-ops',
        ]);
    });

    it('excludes case-insensitively', () => {
        expect(getAutocompleteSuggestions('dev', allTags, ['DEVOPS'])).toEqual([
            'dev-tools',
            'DEV-ops',
        ]);
    });

    it('returns all non-excluded tags for empty query', () => {
        expect(getAutocompleteSuggestions('', allTags, ['prod'])).toEqual([
            'devops',
            'dev-tools',
            'security',
            'DEV-ops',
        ]);
    });

    it('returns empty array when nothing matches', () => {
        expect(getAutocompleteSuggestions('zzz', allTags)).toEqual([]);
    });

    it('returns all tags when query is empty and no exclusions', () => {
        expect(getAutocompleteSuggestions('', allTags)).toEqual(allTags);
    });
});
