/**
 * Tag utilities — pure functions for parsing, formatting, and suggesting tags.
 *
 * Tags are stored as comma-separated strings (e.g. "devops, security, prod").
 * These helpers centralise the split/join/filter logic used across the builder
 * and dashboard contexts.
 */

/**
 * Split a comma-separated tag string into a deduplicated, trimmed array.
 *
 * - Splits on comma
 * - Trims whitespace from each segment
 * - Removes empty strings
 * - Deduplicates (first occurrence wins)
 *
 * @example parseTags("devops, security, devops") // => ["devops", "security"]
 * @example parseTags("")                         // => []
 * @example parseTags("  single  ")               // => ["single"]
 */
export function parseTags(tagString: string): string[] {
    const seen = new Set<string>();
    const result: string[] = [];
    for (const raw of tagString.split(',')) {
        const tag = raw.trim();
        if (tag && !seen.has(tag)) {
            seen.add(tag);
            result.push(tag);
        }
    }
    return result;
}

/**
 * Join an array of tags into a comma-separated string.
 *
 * @example formatTags(["devops", "security"]) // => "devops, security"
 * @example formatTags([])                     // => ""
 */
export function formatTags(tags: string[]): string {
    return tags.join(', ');
}

/**
 * Return autocomplete suggestions for a query prefix.
 *
 * - Case-insensitive prefix match
 * - Excludes tags listed in `excludeTags`
 * - Empty query returns all non-excluded tags
 * - Results preserve the order of `allTags`
 *
 * @example
 * getAutocompleteSuggestions("dev", ["devops", "security", "dev-tools"], ["devops"])
 * // => ["dev-tools"]
 */
export function getAutocompleteSuggestions(
    query: string,
    allTags: string[],
    excludeTags: string[] = [],
): string[] {
    const lowerQuery = query.toLowerCase();
    const excludeSet = new Set(excludeTags.map((t) => t.toLowerCase()));
    return allTags.filter((tag) => {
        const lower = tag.toLowerCase();
        return lower.startsWith(lowerQuery) && !excludeSet.has(lower);
    });
}
