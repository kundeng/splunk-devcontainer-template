import { useEffect, useRef } from 'react';

/**
 * Registers a beforeunload listener when isDirty is true.
 * Shows browser's native "unsaved changes" dialog on page close/refresh.
 */
export function useUnsavedGuard(isDirty: boolean): void {
    const dirtyRef = useRef(isDirty);
    dirtyRef.current = isDirty;

    useEffect(() => {
        const handler = (e: BeforeUnloadEvent) => {
            if (dirtyRef.current) {
                e.preventDefault();
            }
        };

        window.addEventListener('beforeunload', handler);
        return () => window.removeEventListener('beforeunload', handler);
    }, []);
}
